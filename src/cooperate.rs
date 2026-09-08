//! Small, process-wide cooperation primitives for tokenizer work.
//!
//! The gate limits active quanta across every tokenizer instance in this
//! process. It deliberately measures CPU capacity, not external system load:
//! the standard library has no portable load signal.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc::{self, Receiver, SyncSender},
    Arc, Condvar, Mutex, MutexGuard, OnceLock,
};
use std::thread;
use std::time::{Duration, Instant};

const MIN_QUANTUM: usize = 16 * 1024;
const MAX_QUANTUM: usize = 256 * 1024;
const INITIAL_QUANTUM: usize = 64 * 1024;
const TARGET_NANOS: u128 = 1_000_000;
const REFRESH_INTERVAL: Duration = Duration::from_secs(1);
const WAIT_INTERVAL: Duration = Duration::from_millis(250);
type Job = Box<dyn FnOnce() + Send + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub detected_cpus: usize,
    pub worker_limit: usize,
    pub active: usize,
    pub peak_active: usize,
    pub completed_quanta: usize,
    pub waits: usize,
    pub yields: usize,
    pub cooldowns: usize,
    pub pool_threads: usize,
    pub submitted_jobs: usize,
    pub inline_fallbacks: usize,
}

struct GateState {
    detected_cpus: usize,
    worker_limit: usize,
    active: usize,
    peak_active: usize,
    completed_quanta: usize,
    waits: usize,
    waiting: usize,
    yields: usize,
    cooldowns: usize,
    quantum: usize,
    last_refresh: Instant,
    rest_started: Option<Instant>,
    rest_debt_ns: i128,
    last_accounted: Instant,
    busy_since_yield_ns: u128,
    #[cfg(test)]
    forced_cpus: Option<usize>,
}

struct Gate {
    state: Mutex<GateState>,
    changed: Condvar,
}

static GATE: OnceLock<Gate> = OnceLock::new();
static POOL: OnceLock<Option<SharedPool>> = OnceLock::new();
static POOL_THREADS: AtomicUsize = AtomicUsize::new(0);
static SUBMITTED_JOBS: AtomicUsize = AtomicUsize::new(0);
static INLINE_FALLBACKS: AtomicUsize = AtomicUsize::new(0);

struct SharedPool {
    sender: SyncSender<Job>,
}

fn lock<'a, T>(mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

fn cap_for_cpus(cpus: usize) -> usize {
    (cpus / 32).clamp(1, 8)
}

fn detected_cpus() -> usize {
    thread::available_parallelism().map(|count| count.get()).unwrap_or(1).max(1)
}

fn initial_state() -> GateState {
    GateState {
        detected_cpus: 1,
        worker_limit: 1,
        active: 0,
        peak_active: 0,
        completed_quanta: 0,
        waits: 0,
        waiting: 0,
        yields: 0,
        cooldowns: 0,
        quantum: INITIAL_QUANTUM,
        last_refresh: Instant::now().checked_sub(REFRESH_INTERVAL).unwrap_or_else(Instant::now),
        rest_started: None,
        rest_debt_ns: 0,
        last_accounted: Instant::now(),
        busy_since_yield_ns: 0,
        #[cfg(test)]
        forced_cpus: None,
    }
}

fn gate() -> &'static Gate {
    GATE.get_or_init(|| Gate { state: Mutex::new(initial_state()), changed: Condvar::new() })
}

fn refresh(state: &mut GateState, now: Instant) {
    if now.saturating_duration_since(state.last_refresh) < REFRESH_INTERVAL {
        return;
    }
    #[cfg(test)]
    let cpus = state.forced_cpus.unwrap_or_else(detected_cpus);
    #[cfg(not(test))]
    let cpus = detected_cpus();
    state.detected_cpus = cpus;
    state.worker_limit = cap_for_cpus(cpus);

    state.last_refresh = now;
    state.quantum = state.quantum.clamp(MIN_QUANTUM, MAX_QUANTUM);
}

fn wait_for<'a>(changed: &Condvar, state: MutexGuard<'a, GateState>, duration: Duration) -> MutexGuard<'a, GateState> {
    match changed.wait_timeout(state, duration) {
        Ok((state, _)) => state,
        Err(error) => error.into_inner().0,
    }
}

fn cooldown_delay(detected_cpus: usize, waiting: usize, elapsed: Duration) -> Option<Duration> {
    if detected_cpus == 1 && !elapsed.is_zero() {
        Some(elapsed)
    } else if waiting > 0 && !elapsed.is_zero() {
        // ponytail: contention uses a 50% rest duty; tune from foreground measurements.
        Some(elapsed)
    } else {
        None
    }
}

// Account for actual OS rest, including coarse timer overshoot. Negative debt
// permits later work instead of requesting another sub-millisecond sleep.
fn remaining_rest(state: &mut GateState, now: Instant) -> Option<Duration> {
    let started = state.rest_started?;
    let elapsed = now.saturating_duration_since(started).as_nanos().min(i128::MAX as u128) as i128;
    if elapsed >= state.rest_debt_ns {
        // ponytail: credit at most 20 ms; long idle periods must not remove future cooperation.
        state.rest_debt_ns = state.rest_debt_ns.saturating_sub(elapsed).max(-20_000_000);
        state.rest_started = None;
        None
    } else {
        Some(Duration::from_nanos((state.rest_debt_ns - elapsed).min(u64::MAX as i128) as u64))
    }
}

fn account_busy(state: &mut GateState, now: Instant) {
    let elapsed = now.saturating_duration_since(state.last_accounted);
    state.last_accounted = now;
    // Charge shared wall time once, even when several quanta overlap.
    if state.active > 0 {
        state.busy_since_yield_ns = state.busy_since_yield_ns.saturating_add(elapsed.as_nanos());
        if let Some(delay) = cooldown_delay(state.detected_cpus, state.waiting, elapsed) {
            state.rest_debt_ns = state.rest_debt_ns.saturating_add(delay.as_nanos().min(i128::MAX as u128) as i128);
        }
    }
}

fn release_slot(state: &mut GateState, now: Instant) -> bool {
    account_busy(state, now);
    state.active = state.active.saturating_sub(1);
    // Drain existing quanta before measuring global rest; active work is not idle time.
    if state.active == 0 && state.rest_debt_ns > 0 {
        state.rest_started = Some(now);
        state.cooldowns = state.cooldowns.saturating_add(1);
    }
    let yield_now = state.waiting > 0 || state.rest_started.is_some() || state.busy_since_yield_ns >= TARGET_NANOS;
    if yield_now {
        state.busy_since_yield_ns = 0;
        state.yields = state.yields.saturating_add(1);
    }
    yield_now
}

fn next_quantum(current: usize, bytes: usize, elapsed: Duration) -> usize {
    if bytes == 0 {
        return current.clamp(MIN_QUANTUM, MAX_QUANTUM);
    }
    let nanos = elapsed.as_nanos().max(1);
    let observed = (bytes as u128)
        .saturating_mul(TARGET_NANOS)
        .checked_div(nanos)
        .unwrap_or(MIN_QUANTUM as u128)
        .min(usize::MAX as u128) as usize;
    let smoothed = current.saturating_mul(3).saturating_add(observed) / 4;
    smoothed.clamp(MIN_QUANTUM, MAX_QUANTUM)
}

pub(crate) fn acquire() -> Permit<'static> {
    acquire_from(gate())
}

fn acquire_from<'a>(gate: &'a Gate) -> Permit<'a> {
    let mut state = lock(&gate.state);
    loop {
        let now = Instant::now();
        refresh(&mut state, now);
        account_busy(&mut state, now);
        let cooldown = remaining_rest(&mut state, now);
        if state.active < state.worker_limit && state.rest_debt_ns <= 0 {
            state.active += 1;
            state.peak_active = state.peak_active.max(state.active);
            return Permit { gate, started: now, finished: false, byte_budget: state.quantum };
        }

        state.waits += 1;
        let wait = cooldown.unwrap_or(WAIT_INTERVAL).min(WAIT_INTERVAL);
        state.waiting += 1;
        state = wait_for(&gate.changed, state, wait);
        account_busy(&mut state, Instant::now());
        state.waiting -= 1;
    }
}

pub(crate) struct Permit<'a> {
    gate: &'a Gate,
    started: Instant,
    finished: bool,
    byte_budget: usize,
}

impl<'a> Permit<'a> {
    pub(crate) fn byte_budget(&self) -> usize { self.byte_budget }

    pub(crate) fn finish(mut self, bytes: usize) {
        self.finished = true;
        let gate = self.gate;
        let mut state = lock(&gate.state);
        let now = Instant::now();
        refresh(&mut state, now);
        state.completed_quanta = state.completed_quanta.saturating_add(1);
        // Tiny partial quanta measure call overhead, not sustained byte throughput.
        if bytes >= MIN_QUANTUM { state.quantum = next_quantum(state.quantum, bytes, now.saturating_duration_since(self.started)); }
        let yield_now = release_slot(&mut state, now);
        if state.waiting > 0 { gate.changed.notify_one(); }
        drop(state);
        if yield_now { thread::yield_now(); }
    }
}

impl<'a> Drop for Permit<'a> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let gate = self.gate;
        let mut state = lock(&gate.state);
        let now = Instant::now();
        refresh(&mut state, now);
        let yield_now = release_slot(&mut state, now);
        if state.waiting > 0 { gate.changed.notify_one(); }
        drop(state);
        if yield_now { thread::yield_now(); }
    }
}

pub(crate) fn worker_limit() -> usize {
    let gate = gate();
    let mut state = lock(&gate.state);
    refresh(&mut state, Instant::now());
    state.worker_limit
}

fn stats_for(gate: &Gate) -> Stats {
    let mut state = lock(&gate.state);
    refresh(&mut state, Instant::now());
    Stats {
        detected_cpus: state.detected_cpus,
        worker_limit: state.worker_limit,
        active: state.active,
        peak_active: state.peak_active,
        completed_quanta: state.completed_quanta,
        waits: state.waits,
        yields: state.yields,
        cooldowns: state.cooldowns,
        pool_threads: POOL_THREADS.load(Ordering::Relaxed),
        submitted_jobs: SUBMITTED_JOBS.load(Ordering::Relaxed),
        inline_fallbacks: INLINE_FALLBACKS.load(Ordering::Relaxed),
    }
}

pub(crate) fn stats() -> Stats {
    stats_for(gate())
}
fn shared_pool() -> Option<&'static SharedPool> {
    POOL.get_or_init(create_pool).as_ref()
}

fn create_pool() -> Option<SharedPool> {
    let workers = worker_limit();
    let (sender, receiver) = mpsc::sync_channel::<Job>(workers.saturating_mul(2).max(1));
    let receiver = Arc::new(Mutex::new(receiver));
    let mut handles = Vec::with_capacity(workers);
    for index in 0..workers {
        let receiver = Arc::clone(&receiver);
        match thread::Builder::new()
            .name(format!("rwkv-cooperate-{index}"))
            .spawn(move || worker_loop(receiver))
        {
            Ok(handle) => {
                handles.push(handle);
                POOL_THREADS.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                drop(sender);
                for handle in handles {
                    let _ = handle.join();
                    POOL_THREADS.fetch_sub(1, Ordering::Relaxed);
                }
                return None;
            }
        }
    }
    Some(SharedPool { sender })
}

fn worker_loop(receiver: Arc<Mutex<Receiver<Job>>>) {
    loop {
        let job = {
            let receiver = lock(&receiver);
            receiver.recv()
        };
        let Ok(job) = job else { break };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    }
}

fn run_inline(job: Job) {
    INLINE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
}

/// Submit work to the process-wide lazy queue. Jobs must not recursively submit
/// more work to this queue; a full queue is intentionally backpressured.
pub(crate) fn submit<F>(job: F)
where
    F: FnOnce() + Send + 'static,
{
    let job: Job = Box::new(job);
    let Some(pool) = shared_pool() else {
        run_inline(job);
        return;
    };
    match pool.sender.send(job) {
        Ok(()) => {
            SUBMITTED_JOBS.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => run_inline(error.0),
    }
}

#[cfg(test)]
fn local_gate(detected_cpus: usize) -> Gate {
    let mut state = initial_state();
    state.forced_cpus = Some(detected_cpus);
    state.last_refresh = Instant::now().checked_sub(REFRESH_INTERVAL).unwrap_or_else(Instant::now);
    Gate { state: Mutex::new(state), changed: Condvar::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn counters_advanced(before: Stats, after: Stats) -> bool {
        after.submitted_jobs > before.submitted_jobs || after.inline_fallbacks > before.inline_fallbacks
    }

    #[test]
    fn cap_function_covers_requested_cpu_counts() {
        assert_eq!(cap_for_cpus(1), 1);
        assert_eq!(cap_for_cpus(2), 1);
        assert_eq!(cap_for_cpus(4), 1);
        assert_eq!(cap_for_cpus(8), 1);
        assert_eq!(cap_for_cpus(32), 1);
        assert_eq!(cap_for_cpus(128), 4);
        assert_eq!(cap_for_cpus(256), 8);
    }

    #[test]
    fn permits_share_one_local_cap_across_callers() {
        for cpus in [1, 2, 4, 8, 32, 128, 256] {
            let gate = local_gate(cpus);
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            thread::scope(|scope| {
                for _ in 0..8 {
                    let active = &active;
                    let peak = &peak;
                    let gate = &gate;
                    scope.spawn(move || {
                        let permit = acquire_from(gate);
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(5));
                        active.fetch_sub(1, Ordering::SeqCst);
                        permit.finish(MIN_QUANTUM);
                    });
                }
            });
            assert!(peak.load(Ordering::SeqCst) <= cap_for_cpus(cpus));
            assert_eq!(stats_for(&gate).active, 0);
            assert_eq!(stats_for(&gate).completed_quanta, 8);
        }
    }

    #[test]
    fn one_cpu_cooldown_is_visible_and_shared() {
        let gate = local_gate(1);
        let before = stats_for(&gate).cooldowns;
        let permit = acquire_from(&gate);
        thread::sleep(Duration::from_millis(1));
        permit.finish(MIN_QUANTUM);
        assert!(stats_for(&gate).cooldowns > before);
        let permit = acquire_from(&gate);
        permit.finish(0);
    }

    #[test]
    fn permit_drop_and_panic_cleanup_release_active_slot() {
        let gate = local_gate(8);
        {
            let permit = acquire_from(&gate);
            drop(permit);
        }
        assert_eq!(stats_for(&gate).active, 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _permit = acquire_from(&gate);
            panic!("test panic");
        }));
        assert!(result.is_err());
        assert_eq!(stats_for(&gate).active, 0);
        let permit = acquire_from(&gate);
        permit.finish(0);
    }

    #[test]
    fn timer_overshoot_is_credited_and_overlapping_work_is_counted_once() {
        let mut state = initial_state();
        state.detected_cpus = 32;
        state.waiting = 1;
        state.active = 2;
        let now = Instant::now();
        state.last_accounted = now;
        let finished = now + Duration::from_millis(4);
        release_slot(&mut state, finished);
        assert_eq!(state.rest_debt_ns, 4_000_000);
        assert!(state.rest_started.is_none());
        release_slot(&mut state, finished);
        assert_eq!(state.rest_debt_ns, 4_000_000, "overlapping permit must not double-charge");
        assert_eq!(remaining_rest(&mut state, finished), Some(Duration::from_millis(4)));
        assert_eq!(remaining_rest(&mut state, finished + Duration::from_millis(16)), None);
        assert_eq!(state.rest_debt_ns, -12_000_000);
        account_busy(&mut state, finished + Duration::from_millis(16));
        state.active = 1;
        release_slot(&mut state, finished + Duration::from_millis(20));
        assert_eq!(state.rest_debt_ns, -8_000_000);
        assert!(state.rest_started.is_none());
    }

    #[test]
    fn tiny_calls_accumulate_busy_time_before_yielding() {
        let mut state = initial_state();
        state.detected_cpus = 32;
        let now = Instant::now();
        state.last_accounted = now;
        state.active = 1;
        assert!(!release_slot(&mut state, now + Duration::from_micros(100)));
        state.active = 1;
        assert!(release_slot(&mut state, now + Duration::from_millis(1)));
        assert_eq!(state.yields, 1);
        assert_eq!(state.busy_since_yield_ns, 0);
    }

    #[test]
    fn queued_callers_get_rest_without_slowing_uncontended_calls() {
        let elapsed = Duration::from_millis(4);
        assert_eq!(cooldown_delay(32, 0, elapsed), None);
        assert_eq!(cooldown_delay(32, 1, elapsed), Some(elapsed));
        assert_eq!(cooldown_delay(1, 0, elapsed), Some(elapsed));
    }

    #[test]
    fn byte_quantum_adjustment_is_smoothed_and_clamped() {
        let up = next_quantum(MIN_QUANTUM, MAX_QUANTUM, Duration::from_micros(10));
        let down = next_quantum(MAX_QUANTUM, MIN_QUANTUM, Duration::from_millis(100));
        assert!(up > MIN_QUANTUM && up <= MAX_QUANTUM);
        assert!(down < MAX_QUANTUM && down >= MIN_QUANTUM);
        assert_eq!(next_quantum(64 * 1024, 0, Duration::ZERO), 64 * 1024);
    }

    #[test]
    fn shared_queue_reuses_workers_after_a_panicking_job() {
        let (jobs, receiver) = mpsc::sync_channel::<Job>(2);
        let worker = thread::spawn(move || worker_loop(Arc::new(Mutex::new(receiver))));
        let (done_tx, done_rx) = mpsc::channel();
        jobs.send(Box::new(|| panic!("private queue job panic"))).unwrap();
        jobs.send(Box::new(move || done_tx.send(()).unwrap())).unwrap();
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(jobs);
        worker.join().unwrap();
    }

    #[test]
    fn counters_have_a_positive_control_against_a_noop_snapshot() {
        let before = stats();
        assert!(!counters_advanced(before, before));
        submit(|| {});
        let after = stats();
        assert!(counters_advanced(before, after));
    }
}