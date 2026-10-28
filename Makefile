PYTHON ?= python
REPO ?= ../more-than-a-hack-2027

.PHONY: submission test validate

submission:
	$(PYTHON) -m possession.make_submission --out outputs/submission.csv

validate: submission
	$(PYTHON) $(REPO)/evaluation/validate_submission.py outputs/submission.csv

test:
	$(PYTHON) -m unittest discover -s tests -v
