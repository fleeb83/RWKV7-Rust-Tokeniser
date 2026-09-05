import argparse
import math
import unittest

from run_duration import collect_window, run, timing_samples


class SteppingClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        value = self.value
        self.value += 1
        return value


def rust_row():
    return {"case_id": "case", "token_ids": [1], "decoded_bytes": [97], "encode_nanos": [10], "decode_nanos": [2]}


class DurationTests(unittest.TestCase):
    def test_clock_controlled_window_retains_per_case_samples(self):
        labels = []

        def child(label):
            labels.append(label)
            return {"cases": [rust_row()]}

        final, encode, decode, cycles, completed, elapsed = collect_window(2, 1, child, SteppingClock())
        self.assertEqual(labels, ["warmup-1", "cycle-0"])
        self.assertEqual((cycles, completed, elapsed), (1, 1, 3))
        self.assertEqual(encode, [10])
        self.assertEqual(decode, [2])
        self.assertEqual(final["case"]["timings_ns"], {"encode": [10], "decode": [2]})

    def test_python_child_schema_is_normalised(self):
        encode, decode = timing_samples(
            {"case_id": "python", "timings_ns": {"encode": [11], "decode": [3]}}
        )
        self.assertEqual((encode, decode), ([11], [3]))

    def test_nested_observed_row_and_identity_are_retained(self):
        row = {
            "case_id": "python",
            "observed": {"token_ids": [7], "decoded_bytes": [195, 169]},
            "timings_ns": {"encode": [11], "decode": [3]},
        }
        final = {}
        encode = []
        decode = []
        from run_duration import merge_cycle_rows

        merge_cycle_rows([row], final, encode, decode, 1)
        self.assertEqual(final["python"]["observed"], {"token_ids": [7], "decoded_bytes": [195, 169]})
        self.assertEqual(final["python"]["timings_ns"], {"encode": [11], "decode": [3]})

    def test_missing_or_invalid_samples_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "missing timing samples"):
            timing_samples({"case_id": "missing"})
        with self.assertRaisesRegex(RuntimeError, "invalid timing sample"):
            timing_samples({"case_id": "bad", "encode_nanos": [True], "decode_nanos": [1]})
        with self.assertRaisesRegex(RuntimeError, "unequal encode/decode"):
            timing_samples({"case_id": "unequal", "encode_nanos": [1, 2], "decode_nanos": [1]})

    def test_nonfinite_duration_is_rejected_without_inputs_or_child(self):
        args = argparse.Namespace(duration_seconds=math.nan, warmups=1)
        with self.assertRaisesRegex(ValueError, "duration-seconds"):
            run(args)


if __name__ == "__main__":
    unittest.main()
