PYTHON ?= .venv/bin/python
REPO ?= ../more-than-a-hack-2027
N_JOBS ?= 2
SEEDS ?=
CELLS ?=

.PHONY: submission test validate freeze-check headline

submission:
	$(PYTHON) -m possession.make_submission --out outputs/submission.csv

validate: submission
	$(PYTHON) $(REPO)/evaluation/validate_submission.py outputs/submission.csv

test:
	$(PYTHON) scripts/constant_provenance.py --check
	$(PYTHON) -m unittest discover -s tests -v

# One command that asserts the D-029 freeze end to end; fails loudly, never repairs.
freeze-check:
	$(PYTHON) scripts/freeze_check.py

# One command that reproduces the frozen dev-v1 headline (D-029, B01/Q28): master .4247 vs ship
# .4039 = +.0208 [+.0169,+.0250], 0/10, and master - baseline_plus +.0988. ~71 min at N_JOBS=2.
# Smoke: `make headline SEEDS=0 CELLS=1:0.05` (one seed, one cell; the log says it is not the
# frozen protocol). The result file is scratch/B16/headline.json, outside the repository.
headline:
	$(PYTHON) scripts/headline.py --n-jobs $(N_JOBS) $(if $(SEEDS),--seeds "$(SEEDS)") $(if $(CELLS),--cells "$(CELLS)")
