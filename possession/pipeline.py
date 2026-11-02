"""The shipped architecture (D-028), in one deterministic entry point.

Fixed stage order -- do not reorder, do not add stages here:

    structure -> baseline_plus.predict (r 2.5, k 16, m 20)
              -> dgate.apply   (whole ball-missing gap >= 2 s, mean P(D) > .5)
              -> growd.apply   (R 25 m, cap 30 s)
              -> ingap.apply   (ball-free HMM, LAM 3.88)
              -> subgap.apply  (Q .5 / TF 50 / T2 50)
              -> postprocess.apply(labels, postprocess.runs_from_half(match.half))   # floor, last label write

The short-spell floor (E14) is applied **after** the flag-gated block, on the single return path, so
the gate-off (Baseline+) and add-on-off (B04) configurations get it too; staging it is
``postprocess.ENABLED = False`` / ``apply(..., enable=False)``.

Not imported here by construction (D-028): ``anchor``, ``hsmm``, ``supervised``,
``spells_from_events``, ``touches``, ``gk_possession``, ``deadball``. The touch/touch-decoder arm is
closed; E07's dead-ball sharpeners fold in via B02 as tier 2 only.

``models`` lets the dev harness inject leave-one-game-out models:
``(dgate_model, dgate_columns, ballfree_model)``. Left as ``None`` the build trains on the external
games only (:func:`possession.dgate.build_model`, :func:`ballfree_model`), never on the match.

No match labels, no public GT and no network are read. Deterministic: fixed seeds in the two models,
no sampling anywhere else.
"""
from possession import baseline_plus, dgate, growd, ingap, postprocess, structure, subgap
from possession.data import load_match


def ballfree_model():
    """Ball-free P(H) model trained on the external games only (never on the match)."""
    from possession import ballfree, external
    tabs = []
    for gid in dgate.training_games():
        _, X, Xs, y = ballfree.game_table(external.load_game(gid))
        k = y >= 0
        tabs.append((X[k], Xs[k], y[k]))
    return ballfree.fit(tabs, "hgb")


def ingap_ph(match, model=None, pitch=None):
    """Per-frame ball-free P(H) for ``match``; ``model`` is a pre-fitted ballfree model."""
    from possession import ballfree
    if model is None:
        model = ballfree_model()
    return ballfree.predict_frames(model, match, pitch=pitch or (structure.PITCH_L, structure.PITCH_W))


def build_labels(match=None, models=None, log=print, guard=True, forced=None, pitch=None):
    """Run the D-028 stack once and return ``(N,)`` H/A/D/X labels.

    ``match`` defaults to the shipped match (``load_match()``); ``models`` is the dev-harness
    ``(dgate_model, dgate_columns, ballfree_model)`` triple, already trained leave-one-game-out.
    ``log`` receives one line per stage footprint (label-free guards); pass ``None`` to silence.
    ``guard=False`` keeps the identical label path but skips the BAR-ATH-calibrated build canaries,
    which the dev harness must do (their bounds are not meaningful on degraded practice games).
    ``forced`` overrides the published pre-kickoff D mask; the dev harness passes all-False because
    D-007's blocks are BAR-ATH structure, not a property of the practice games. ``pitch`` likewise
    lets the dev harness use the external game's own pitch instead of BAR-ATH's 105.3 x 68.
    """
    match = load_match() if match is None else match
    say = (lambda *a: None) if log is None else log

    # 1. Organizer-published structure (halves, kickoffs, forced-D block; BAR-ATH constants).
    struct = structure.structure(match)

    # 2. Baseline+: persistent last owner, hysteresis, out-of-bounds D, spell merge.
    labels = baseline_plus.predict(match, forced=struct.forced_dead_mask if forced is None else forced,
                                   pitch=pitch)

    # 3-6. The flag-gated D/add-on stack; ingap is only safe on D-gated labels (Q17).
    if dgate.ENABLED:
        dmodel, cols, bf = models if models is not None else (None, None, None)
        if dmodel is None:
            dmodel, cols = dgate.build_model()
        labels, dead, fp = dgate.apply(labels, match, model=(dmodel, cols), check=guard)
        say("dgate:", fp)
        gated = labels

        labels, fp = growd.apply(gated, match, dead=dead, check=guard)
        if fp:
            say("growd:", fp)

        if ingap.ENABLED:
            grown = labels
            labels = ingap.apply(grown, match.B, match.half, ingap_ph(match, bf, pitch), dead=dead,
                                 check=guard)
            say("ingap:", ingap.footprint(grown, labels, match.B, match.half, dead))

        if subgap.ENABLED:
            pf = dgate.frame_pd(dmodel, cols, match.P, match.V, match.T, match.half)
            labels, fp = subgap.apply(labels, gated, match, pf, fired=dead, check=guard)
            say("subgap:", fp)

    # 7. E14 short-spell floor (1.5 s -> D): the last label write, on every return path.
    #    `postprocess.ENABLED` is read here, so the flag (or enable=False) stages it for B04-style checks.
    labels = postprocess.apply(labels, postprocess.runs_from_half(match.half),
                               enable=postprocess.ENABLED)
    return labels
