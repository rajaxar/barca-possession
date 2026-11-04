PYTHON ?= .venv/bin/python
REPO ?= ../more-than-a-hack-2027
N_JOBS ?= 2
SEEDS ?=
CELLS ?=

.PHONY: submission test validate freeze-check headline notebook-local

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

# One command that refreshes the personal, local-only HTML reading copy. It reuses the frozen CSV
# whenever numbers.json --check is green, so a parallel card touching an unrelated file cannot force a
# rebuild; it rebuilds only when the CSV is missing or the manifest reports drift, verifies the frozen
# sha, then refreshes numbers, figures, the local notebook and the local page. Never commit or publish
# the output (D-002/D-026): notebooks/possession_walkthrough_local.html is gitignored.
notebook-local:
	@set -e; \
	if [ ! -f outputs/submission.csv ]; then \
	  echo "notebook-local: submission.csv missing -> rebuilding"; \
	  $(PYTHON) -m possession.make_submission --out outputs/submission.csv; \
	  $(PYTHON) scripts/walkthrough_numbers.py; \
	elif $(PYTHON) scripts/walkthrough_numbers.py --check; then \
	  echo "notebook-local: submission.csv reused (numbers.json --check green)"; \
	else \
	  echo "notebook-local: numbers.json --check reports drift -> rebuilding submission.csv"; \
	  $(PYTHON) -m possession.make_submission --out outputs/submission.csv; \
	  $(PYTHON) scripts/walkthrough_numbers.py; \
	fi; \
	sha=$$(shasum -a 256 outputs/submission.csv | cut -d' ' -f1); \
	if [ "$$sha" != "ee7b5b6bdf7404e82696654ce01b9d823e5b54f149028dcfdeaa5626cf3646d8" ]; then \
	  echo "notebook-local: submission.csv sha mismatch"; \
	  echo "  expected ee7b5b6bdf7404e82696654ce01b9d823e5b54f149028dcfdeaa5626cf3646d8"; \
	  echo "  actual   $$sha"; \
	  exit 1; \
	fi
	$(PYTHON) scripts/walkthrough_numbers.py
	$(PYTHON) scripts/walkthrough_numbers.py --check
	$(PYTHON) scripts/build_walkthrough.py --profile local
	$(PYTHON) scripts/build_walkthrough_html.py --profile local
	@$(PYTHON) -c 'import hashlib, os, re; p = "notebooks/possession_walkthrough_local.html"; d = open(p, encoding="utf-8").read(); s = os.path.getsize(p); n = len(re.findall(r"data:image/", d)); h = hashlib.sha256(open("outputs/submission.csv", "rb").read()).hexdigest(); print("notebook-local: %s" % p); print("  size: %d bytes (%.2f MiB)" % (s, s / 1048576.0)); print("  inlined figures: %d" % n); print("  CSV sha256: %s" % h); print("  numbers.json: checked green (--check above)")'
