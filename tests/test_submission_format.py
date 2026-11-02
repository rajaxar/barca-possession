"""Run the organizer's validator on the generated submission."""
import os
import subprocess
import sys
import tempfile
import unittest

from possession import config, make_submission


class SubmissionFormat(unittest.TestCase):
    def test_validator_accepts_output(self):
        validator = os.path.join(config.repo_dir(), "evaluation", "validate_submission.py")
        if not os.path.isfile(validator):
            self.skipTest("organizer validator not found at {} (set MTAH_REPO_DIR)".format(validator))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "submission.csv")
            make_submission.main(["--out", out])
            res = subprocess.run([sys.executable, validator, out], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
