PYTHON ?= .venv/bin/python
REPO ?= ../more-than-a-hack-2027

.PHONY: submission test validate freeze-check

submission:
	$(PYTHON) -m possession.make_submission --out outputs/submission.csv

validate: submission
	$(PYTHON) $(REPO)/evaluation/validate_submission.py outputs/submission.csv

test:
	$(PYTHON) -m unittest discover -s tests -v

# One command that asserts the D-029 freeze end to end; fails loudly, never repairs.
freeze-check:
	$(PYTHON) scripts/freeze_check.py
