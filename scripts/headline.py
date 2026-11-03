"""Reproduce the frozen dev-v1 headline in one command: `master` vs `ship`, on the D-029 artifact.

Anchors: work/DECISIONS.md D-022 (the frozen grid) and D-029 (the frozen artifact and its numbers),
work/cards/B16-make-headline.md, research/B01_result.md, research/Q28_headline_review.md.

    cd solution && make headline                      # the full frozen protocol, ~71 min at n_jobs=2
    cd solution && make headline SEEDS=0 CELLS=1:0.05  # smoke: one seed, one cell, ~2 min

What it does, and does not do:

* Runs the **unmodified** frozen harness: ``dev_v1._run_game`` per game (the same worker
  ``dev_v1.run_dev_v1`` forks) over ``dev_v1.GAMES``, ``dev_v1.SEEDS`` and ``dev_v1.CELLS``, with
  ``master``, ``ship``, ``baseline_plus`` and ``smooth50`` sharing each cell and its memo. Nothing in
  ``possession/`` is imported differently, imported twice or patched; the seeds/cells flags only
  narrow the module-level tuples the harness already reads (a smoke run is not the frozen protocol
  and says so in the log and the result file).
* Adds per-game checkpoints under ``scratch/B16/parts/<fingerprint>/``: a killed run resumes the
  games it does not have. The fingerprint covers the pipelines, seeds, cells, the git commit, the
  working-tree state and a hash of every ``possession/*.py``, so a code change cannot silently
  reuse a stale part.
* Prints the reference table (levels, deltas, game-cluster CIs, games-negative) and the ``master``
  vs ``ship`` / ``baseline_plus`` group, shift and per-cell rows, then checks the D-029 numbers.
  A mismatch is reported as a discrepancy and exits non-zero; **nothing is adjusted or re-run to
  make it fit**.
* Prints the frozen CSV sha256 and the git commit so a run is self-describing, and writes the whole
  result to ``scratch/B16/headline.json`` — outside the solution repository (GUARDRAILS §4:
  throwaway output belongs in ``scratch/<card-id>/``), so it cannot be committed.

``public_gt: none``: no scorer, no public answers, no match labels. The 10 external practice games
only (GUARDRAILS §2).
"""
import argparse
import collections
import datetime
import hashlib
import json
import multiprocessing
import os
import pickle
import subprocess
import sys
import time

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SOLUTION)
SCRATCH = os.path.join(ROOT, "scratch", "B16")

PIPELINES = ("master", "ship", "baseline_plus", "smooth50")

# The four-decimal reference this run exists to reproduce (D-029 §"Dev number", B01 §3, Q28 §1-2).
FROZEN_SHA = "ee7b5b6bdf7404e82696654ce01b9d823e5b54f149028dcfdeaa5626cf3646d8"
REFERENCE = collections.OrderedDict([
    ("master", .4247), ("ship", .4039), ("baseline_plus", .3259), ("smooth50", .3037),
])
REFERENCE_DELTAS = collections.OrderedDict([
    # name: (delta, ci_lo, ci_hi, games_negative)
    ("master - ship", (.0208, .0169, .0250, 0)),
    ("master - baseline_plus", (.0988, .0909, .1068, 0)),
])
TOLERANCE = 5e-5           # five decimal places: the 4-dp printed value, not a near miss


class Discrepancy(Exception):
    """A checked number differs from the frozen reference; the message is the finding."""


class Log:
    """Print and append to a log file, flushing each line (a 71-minute run must leave a trace)."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, line=""):
        line = str(line)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def short(sha):
    return sha[:12] if sha else "(none)"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, check=False):
    proc = subprocess.run(["git", "-C", SOLUTION] + list(args), capture_output=True, text=True)
    if check and proc.returncode != 0:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_identity():
    """(full commit, short, [(status lines)], tag commit or None) for the tree being measured."""
    head = git("rev-parse", "HEAD")
    if not head:
        return None, None, [], None
    dirty = [p for p in (git("status", "--porcelain") or "").splitlines() if p.strip()]
    tag = git("rev-parse", "--verify", "final-candidate^{commit}")
    return head, head[:7], dirty, tag


def possession_fingerprint():
    """sha256 over every possession/*.py: the label path and the harness, as bytes on disk."""
    h = hashlib.sha256()
    paths = []
    for base, dirs, files in os.walk(os.path.join(SOLUTION, "possession")):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        paths += [os.path.join(base, f) for f in files if f.endswith(".py")]
    for path in sorted(paths):
        h.update(os.path.relpath(path, SOLUTION).encode())
        with open(path, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest(), len(paths)


def csv_identity(path):
    if not os.path.isfile(path):
        return None, None, None
    rows = 0
    with open(path, "rb") as fh:
        for _ in fh:
            rows += 1
    return sha256_file(path), rows - 1, os.path.getmtime(path)


def parse_seeds(text):
    allowed = (0, 1, 2)
    seeds = [int(s) for s in str(text).replace(" ", "").split(",") if s]
    if not seeds:
        raise SystemExit("--seeds is empty; give a comma list drawn from 0,1,2")
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise SystemExit("--seeds %s: %s is not a frozen dev-v1 seed (D-022 freezes 0,1,2). "
                         "A new degrade seed is a new sample, not a reproduction."
                         % (text, bad))
    return tuple(dict.fromkeys(seeds))


def parse_cells(text, groups, frozen_cells):
    """Named group (primary / stress / v0 / all) or comma-separated knob:v pairs, e.g. 0:0.05,1:0.23."""
    text = str(text).strip()
    key = text.lower()
    if key == "all":
        return tuple(frozen_cells)
    if key in ("v0", "v0_reference"):
        key = "v0_reference"
    if key in groups:
        return tuple(groups[key])
    cells = []
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        try:
            knob, v = item.split(":")
            knob, v = float(knob), float(v)
        except ValueError:
            raise SystemExit("--cells %r: use a group name (%s) or knob:v pairs like 0:0.05"
                             % (text, ", ".join(list(groups) + ["all"])))
        cell = (knob, v)
        if cell not in frozen_cells:
            raise SystemExit("--cells %r: (%g, %g) is not a frozen dev-v1 cell (D-022)"
                             % (text, knob, v))
        cells.append(cell)
    if not cells:
        raise SystemExit("--cells %r selected no cells" % text)
    return tuple(dict.fromkeys(cells))


def fmt(x):
    return "%+.4f" % x


def pct_ci(ci):
    return "[%+.4f, %+.4f]" % (ci[0], ci[1])


def cell_label(knob, v):
    return "knob %g v %-4g" % (knob, v)


def cell_key(knob, v):
    return "k%g_v%g" % (knob, v)


# -- the run ------------------------------------------------------------------------------------

def _checkpoint_worker(job):
    """One game through the frozen per-game worker, checkpointed atomically. Top-level for fork."""
    gi, path = job
    from possession import dev_v1  # inherits the parent's _FNS/_CLEAN via fork
    part = dev_v1._run_game(gi)
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "wb") as fh:
        pickle.dump(part, fh)
    os.replace(tmp, path)
    return gi


def fingerprint_dir(fingerprint):
    """A directory keyed by the hash of the *whole* fingerprint (a prefix would collide)."""
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()
    return os.path.join(SCRATCH, "parts", "fp-" + digest[:16]), digest


def collect(dev_v1, fingerprint, n_jobs, fresh, log):
    """{gi: per-game part} for all 10 games, running only the missing ones; mirrors run_dev_v1's loop."""
    parts_dir, digest = fingerprint_dir(fingerprint)
    os.makedirs(parts_dir, exist_ok=True)
    with open(os.path.join(parts_dir, "fingerprint.txt"), "w", encoding="utf-8") as fh:
        fh.write(fingerprint + "\n")

    def path(gi):
        return os.path.join(parts_dir, "game_%d.pkl" % gi)

    parts, todo = {}, []
    for gi in range(len(dev_v1.GAMES)):
        if not fresh and os.path.isfile(path(gi)):
            try:
                with open(path(gi), "rb") as fh:
                    parts[gi] = pickle.load(fh)
                continue
            except Exception as error:  # noqa: BLE001 - a corrupt checkpoint is simply re-run
                log("  note  checkpoint game %d unreadable (%s); re-running it" % (gi, error))
                os.remove(path(gi))
        todo.append(gi)

    resumed = len(parts)
    log("  checkpoint  %s (fingerprint %s)" % (
        os.path.relpath(parts_dir, ROOT), digest[:16]))
    log("  games       %d resumed from checkpoint, %d to run" % (resumed, len(todo)))

    started = time.time()
    done = 0
    if todo:
        if n_jobs > 1:
            jobs = [(gi, path(gi)) for gi in todo]
            with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
                for gi in pool.imap_unordered(_checkpoint_worker, jobs, chunksize=1):
                    done += 1
                    log("  ..    game %d/10 %-26s %6.0f s elapsed, %2d to go"
                        % (gi, dev_v1.GAMES[gi], time.time() - started, len(todo) - done))
        else:
            for gi in todo:
                _checkpoint_worker((gi, path(gi)))
                done += 1
                log("  ..    game %d/10 %-26s %6.0f s elapsed, %2d to go"
                    % (gi, dev_v1.GAMES[gi], time.time() - started, len(todo) - done))
        for gi in todo:
            with open(path(gi), "rb") as fh:
                parts[gi] = pickle.load(fh)
    log("  run         %.1f min for %d game(s)" % ((time.time() - started) / 60.0, len(todo)))
    return parts, (time.time() - started)


def assemble(dev_v1, parts, names):
    """Exactly run_dev_v1's final line: merge the per-game dicts into one DevV1Result per pipeline."""
    missing = [gi for gi in range(len(dev_v1.GAMES)) if gi not in parts]
    if missing:
        raise RuntimeError("no checkpoint for game(s) %s" % missing)
    return {n: dev_v1.DevV1Result(n, {k: v for p in parts.values() for k, v in p[n].items()})
            for n in names}


def check_coverage(dev_v1, results, seeds, cells):
    """Every (game, seed, cell, shift) scored for every pipeline; a gap would silently bias a mean."""
    want = {(gi, s, k, v, sh) for gi in range(len(dev_v1.GAMES)) for s in seeds
            for k, v in cells for sh in dev_v1.SHIFT_FRAMES}
    for name, res in results.items():
        got = set(res.raw)
        if got != want:
            raise RuntimeError("%s: %d records, expected %d (missing %d, extra %d). Delete %s to "
                               "rebuild the checkpoints." % (
                                   name, len(got), len(want), len(want - got), len(got - want),
                                   os.path.join(SCRATCH, "parts")))
    return len(want)


# -- reporting ----------------------------------------------------------------------------------

def headline_of(res, cells):
    return float(sum(res.cell_score(k, v) for k, v in cells) / len(cells))


def group_cells(dev_v1, cells):
    """{group: cells present in this run} plus the shift rows, so a subset run prints what it has."""
    out = collections.OrderedDict()
    for group in ("primary", "knob0", "knob1", "stress", "v0_reference"):
        have = [c for c in dev_v1.GROUPS[group] if c in cells]
        out[group] = have or None
    return out


def delta(dev_v1, a, b, cells, shift=0):
    return dev_v1._delta(a, b, cells, shift)  # the frozen formula: pooled F1, shared draws, per-game sign


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reproduce the frozen dev-v1 headline (master vs ship) in one command.")
    ap.add_argument("--seeds", default=",".join(str(s) for s in (0, 1, 2)),
                    help="comma list from 0,1,2 (default: the frozen 0,1,2; a subset is a smoke run)")
    ap.add_argument("--cells", default="all",
                    help="'all' (default, 12 frozen cells), a group (primary/stress/v0) or knob:v pairs")
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="worker processes (default 2; the card's process allowance)")
    ap.add_argument("--out", default=os.path.join(SCRATCH, "headline.json"),
                    help="result file (default: %(default)s, outside the git repo)")
    ap.add_argument("--csv", default=os.path.join(SOLUTION, "outputs", "submission.csv"),
                    help="frozen CSV to hash (default: %(default)s)")
    ap.add_argument("--fresh", action="store_true", help="ignore per-game checkpoints")
    args = ap.parse_args(argv)

    os.makedirs(SCRATCH, exist_ok=True)
    log = Log(os.path.join(SCRATCH, "headline.log"))
    t_start = time.time()
    started_at = datetime.datetime.now().astimezone()
    log("")
    log("make headline -- %s" % started_at.isoformat(timespec="seconds"))

    import possession
    from possession import dev_v1

    path = os.path.abspath(possession.__file__)
    if not path.startswith(SOLUTION + os.sep):
        raise SystemExit("refusing to run: possession is imported from %s, not %s (GUARDRAILS §4)"
                         % (path, SOLUTION))

    seeds = parse_seeds(args.seeds)
    cells = parse_cells(args.cells, dev_v1.GROUPS, dev_v1.CELLS)
    full = seeds == tuple(dev_v1.SEEDS) and set(cells) == set(dev_v1.CELLS)
    mode = ("FULL frozen protocol (D-022): seeds %s x %d cells (6 primary + 4 stress + 2 v=0)"
            % (",".join(map(str, seeds)), len(cells))) if full else (
            "SMOKE -- seeds %s x %d cell(s) %s. These numbers are NOT the D-029 headline and are "
            "not checked against it." % (",".join(map(str, seeds)), len(cells),
                                         ", ".join("knob %g v %g" % c for c in cells)))

    head, head_short, dirty, tag = git_identity()
    fp_hash, fp_files = possession_fingerprint()
    fingerprint = "|".join(["headline-v1", ",".join(PIPELINES), "seeds=" + ",".join(map(str, seeds)),
                            "cells=" + ",".join("%g:%g" % c for c in cells), "git=" + str(head),
                            "dirty=" + str(len(dirty)), "code=" + fp_hash,
                            "py=" + sys.version.split()[0]])
    csv_sha, csv_rows, _ = csv_identity(os.path.abspath(args.csv))
    params = dev_v1.params()

    log("")
    log("== provenance")
    log("  mode        %s" % mode)
    log("  n_jobs      %d" % args.n_jobs)
    log("  git         solution @ %s%s" % (head or "(not a checkout)",
        "" if not head else (" (tag final-candidate -> %s)" % short(tag) if tag and tag != head
                             else " (== tag final-candidate)")))
    log("  worktree    %s" % ("clean" if not dirty else "%d uncommitted path(s): %s" % (
        len(dirty), ", ".join(p.split()[-1] for p in dirty[:6]))))
    log("  python      %s" % sys.version.split()[0])
    log("  possession  %s (asserted under solution/)" % path)
    log("  degrader    master %s, params %s" % (dev_v1.DEGRADER_COMMIT, dict(params)))
    log("  games       %d (%d IDSSE + %d Metrica colleagues), leave-one-game-out models"
        % (len(dev_v1.GAMES), 7, 3))
    log("  code hash   %s over %d possession/*.py" % (fp_hash[:16], fp_files))
    if csv_sha:
        verdict = "matches D-029" if csv_sha == FROZEN_SHA else \
            "DIFFERS FROM D-029 (the freeze is %s; rebuild with `make submission` before quoting " \
            "this run as the frozen artifact)" % short(FROZEN_SHA)
        log("  csv         %s  sha256 %s  %s, %s rows" % (
            os.path.relpath(os.path.abspath(args.csv), SOLUTION), csv_sha, verdict,
            "{:,}".format(csv_rows)))
    else:
        log("  csv         %s is missing (no sha printed; run `make submission` for the artifact)"
            % os.path.relpath(os.path.abspath(args.csv), SOLUTION))

    # Narrow the tuples the frozen harness already reads. This is the only state change, and it is
    # restored below; no harness code is edited or re-implemented.
    saved = (dev_v1.SEEDS, dev_v1.CELLS, dev_v1._FNS, dev_v1._CLEAN)
    dev_v1.SEEDS, dev_v1.CELLS = seeds, cells
    dev_v1._FNS = {n: dev_v1.STAGES[n] for n in PIPELINES}
    dev_v1._CLEAN = False
    try:
        log("")
        log("== run: %s" % "one pass, four pipelines sharing each cell and its memo")
        parts, elapsed = collect(dev_v1, fingerprint, args.n_jobs, args.fresh, log)
        results = assemble(dev_v1, parts, PIPELINES)
        n_records = check_coverage(dev_v1, results, seeds, cells)
        log("  coverage    %d records per pipeline (%d games x %d seeds x %d cells x %d shifts)"
            % (n_records, len(dev_v1.GAMES), len(seeds), len(cells), len(dev_v1.SHIFT_FRAMES)))
        log("  class       %s" % dev_v1.provenance())

        # The frozen headline is the mean over the six *primary* cells: ``dev_v1.summary()`` and
        # ``compare_dev_v1`` both read ``GROUPS["primary"]`` (B01 §3, Q28 §1). A full run scores all
        # twelve frozen cells so the tier-1 stress / v=0 / per-cell rows exist, but the headline levels
        # and the two reference deltas are read over the primary cells only.
        headline_cells = [c for c in dev_v1.GROUPS["primary"] if c in cells] or list(cells)
        levels = collections.OrderedDict((n, headline_of(results[n], headline_cells)) for n in PIPELINES)
        comparisons = collections.OrderedDict(
            [("master - ship", ("master", "ship")),
             ("master - baseline_plus", ("master", "baseline_plus")),
             ("master - smooth50", ("master", "smooth50")),
             ("ship - baseline_plus", ("ship", "baseline_plus"))])
        deltas = collections.OrderedDict(
            (label, delta(dev_v1, results[a], results[b], headline_cells))
            for label, (a, b) in comparisons.items())

        log("")
        log("== reference table%s" % ("" if full else " (SMOKE: subset of the frozen grid)"))
        log("  %-16s %.4f   %s" % ("headline", levels["master"],
                                   "mean of the %d primary cell(s) in this run" % len(headline_cells)))
        for name in PIPELINES:
            log("  %-16s %.4f" % (name, levels[name]))
        log("")
        log("  %-24s %-8s %-22s %s" % ("comparison", "delta", "95% CI (game-cluster)",
                                       "games negative"))
        for label, r in deltas.items():
            log("  %-24s %-8s %-22s %d/%d" % (label, fmt(r["delta"]), pct_ci(r["ci"]),
                                              r["neg"], len(dev_v1.GAMES)))
        log("  (%d shared draws, seed-mean inside each draw, resample the %d games -- dev_v1._delta)"
            % (dev_v1.N_DRAWS, len(dev_v1.GAMES)))

        groups = group_cells(dev_v1, cells)
        for label in ("master - ship", "master - baseline_plus"):
            a, b = comparisons[label]
            log("")
            log("  %s by group" % label)
            for group, have in groups.items():
                if have is None:
                    log("    %-14s (not in this run)" % group)
                else:
                    r = delta(dev_v1, results[a], results[b], have)
                    log("    %-14s %+0.5f  %-20s %d/%d" % (
                        group, r["delta"], pct_ci(r["ci"]), r["neg"], len(dev_v1.GAMES)))
            for sh in dev_v1.SHIFT_FRAMES:
                r = delta(dev_v1, results[a], results[b], headline_cells, sh)
                log("    ref shift %+0.1fs %+0.5f  %-20s %d/%d" % (
                    sh / dev_v1.FPS, r["delta"], pct_ci(r["ci"]), r["neg"], len(dev_v1.GAMES)))
            primary = [c for c in dev_v1.GROUPS["primary"] if c in cells]
            if primary:
                log("    %s, primary cells" % label)
                for knob in (0.0, 1.0):
                    row = [c for c in primary if c[0] == knob]
                    if not row:
                        continue
                    log("      knob %g  %s" % (knob, "  ".join(
                        "v %-4g %+0.5f" % (v, delta(dev_v1, results[a], results[b], [(k, v)])[
                            "delta"]) for k, v in row)))

        tier1 = {}
        if full:
            for label in ("master - ship", "master - baseline_plus"):
                a, b = comparisons[label]
                ra, rb = results[a], results[b]
                d = deltas[label]
                t = {
                    "headline_ci_excludes_0": bool(d["ci"][0] > 0 or d["ci"][1] < 0),
                    "primary_cells_all_ge_0": (groups["primary"] is not None and all(
                        delta(dev_v1, ra, rb, [c])["delta"] >= 0 for c in groups["primary"])),
                    "stress_cells_all_ge_0": (groups["stress"] is not None and all(
                        delta(dev_v1, ra, rb, [c])["delta"] >= 0 for c in groups["stress"])),
                    "wins_at_each_shift": all(
                        delta(dev_v1, ra, rb, headline_cells, sh)["delta"] > 0
                        for sh in dev_v1.SHIFT_FRAMES),
                    "v0_ci_low_ge_-.001": (groups["v0_reference"] is not None and
                                           delta(dev_v1, ra, rb, groups["v0_reference"])["ci"][0]
                                           >= -0.001),
                }
                tier1[label] = t
                log("")
                log("  D-022 tier-1 mechanics, %s: %s" % (
                    label, "  ".join("%s=%s" % (k, "yes" if v else "NO") for k, v in t.items())))

        # -- the reproduction check -------------------------------------------------------------
        log("")
        log("== reference check (D-029 / B01 / Q28)")
        failures = []
        if full:
            for name, want in REFERENCE.items():
                got = levels[name]
                ok = abs(got - want) <= TOLERANCE
                log("  %s %-16s %.4f  (reference %.4f)" % ("ok  " if ok else "FAIL", name, got, want))
                if not ok:
                    failures.append("%s: got %.5f, reference %.4f" % (name, got, want))
            for label, (want_d, want_lo, want_hi, want_neg) in REFERENCE_DELTAS.items():
                r = deltas[label]
                ok = (abs(r["delta"] - want_d) <= TOLERANCE
                      and abs(r["ci"][0] - want_lo) <= TOLERANCE
                      and abs(r["ci"][1] - want_hi) <= TOLERANCE
                      and r["neg"] == want_neg)
                log("  %s %-24s %+.4f %s %d/10  (reference %+.4f [%+.4f, %+.4f] %d/10)" % (
                    "ok  " if ok else "FAIL", label, r["delta"], pct_ci(r["ci"]), r["neg"],
                    want_d, want_lo, want_hi, want_neg))
                if not ok:
                    failures.append("%s: got %+.5f %s %d/10, reference %+.4f [%+.4f, %+.4f] %d/10"
                                    % (label, r["delta"], pct_ci(r["ci"]), r["neg"],
                                       want_d, want_lo, want_hi, want_neg))
            if csv_sha and csv_sha != FROZEN_SHA:
                log("  warn csv sha is not the D-029 artifact; the dev number is still the one "
                    "measured above")
        else:
            log("  skipped: SMOKE mode is not the frozen protocol, so no D-029 number is claimed.")

        # -- result file ------------------------------------------------------------------------
        payload = collections.OrderedDict([
            ("card", "B16"), ("mode", "full" if full else "smoke"), ("mode_text", mode),
            ("started", started_at.isoformat(timespec="seconds")),
            ("finished", datetime.datetime.now().astimezone().isoformat(timespec="seconds")),
            ("elapsed_min", round(elapsed / 60.0, 2)),
            ("wall_min", round((time.time() - t_start) / 60.0, 2)),
            ("n_jobs", args.n_jobs), ("seeds", list(seeds)),
            ("cells", [[k, v] for k, v in cells]), ("pipelines", list(PIPELINES)),
            ("headline_cells", [cell_key(k, v) for k, v in headline_cells]),
            ("levels", {k: round(v, 6) for k, v in levels.items()}),
            ("deltas", {label: dict(r) for label, r in deltas.items()}),
            ("tier1", tier1),
            ("cells_detail", {cell_key(k, v): {n: results[n].cell_score(k, v) for n in PIPELINES}
                              for k, v in cells}),
            ("groups", {g: (None if have is None else
                            [cell_key(k, v) for k, v in have]) for g, have in groups.items()}),
            ("git", {"commit": head, "short": head_short, "tag": tag,
                     "dirty_paths": [p.split()[-1] for p in dirty]}),
            ("csv", {"path": os.path.relpath(os.path.abspath(args.csv), SOLUTION),
                     "sha256": csv_sha, "rows": csv_rows,
                     "matches_d029": (csv_sha == FROZEN_SHA) if csv_sha else None}),
            ("possession", path), ("code_sha256", fp_hash),
            ("degrader", {"commit": dev_v1.DEGRADER_COMMIT, "params": dict(params)}),
            ("records_per_pipeline", n_records),
            ("reference", {k: v for k, v in REFERENCE.items()}),
            ("reference_deltas", {k: list(v) for k, v in REFERENCE_DELTAS.items()}),
            ("checks", {"performed": full, "failures": failures,
                        "passed": full and not failures}),
        ])
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = "%s.tmp%d" % (out, os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        os.replace(tmp, out)
        log("")
        log("  result      %s" % os.path.relpath(out, ROOT))
        log("  csv sha256  %s" % (csv_sha or "(missing)"))
        log("  git commit  %s" % (head or "(not a checkout)"))
        log("  runtime     %.1f min wall at n_jobs=%d (%.1f min computing games)"
            % ((time.time() - t_start) / 60.0, args.n_jobs, elapsed / 60.0))

        if failures:
            log("")
            log("DISCREPANCY: the run does not reproduce D-029. Nothing was adjusted or re-run.")
            for f in failures:
                log("  " + f)
            return 1
        if full:
            log("")
            log("REFERENCE REPRODUCED: master %.4f - ship %.4f = %+.4f %s, %d/10 games negative."
                % (levels["master"], levels["ship"], deltas["master - ship"]["delta"],
                   pct_ci(deltas["master - ship"]["ci"]), deltas["master - ship"]["neg"]))
        else:
            log("")
            log("SMOKE RUN OK: the harness ran %d cell(s); no D-029 claim is made."
                % len(cells))
        return 0
    finally:
        dev_v1.SEEDS, dev_v1.CELLS, dev_v1._FNS, dev_v1._CLEAN = saved
        log.close()


if __name__ == "__main__":
    sys.exit(main())
