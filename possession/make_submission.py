"""Generate the deterministic submission: Baseline+ -> missing-ball D gate (B04) -> optional grown present-ball D (E66)
-> optional in-gap decoder -> optional sub-gap gate (E66).

The gate's P(D) model is trained at build time on the external games only (see :mod:`possession.dgate`).
The in-gap ball-free decoder (D08: on by default) runs strictly after the gate; if the gate is disabled it and both E66
add-ons (:mod:`possession.growd`, :mod:`possession.subgap`; each has its own ``ENABLED`` flag) are skipped too.

Usage: python -m possession.make_submission --out outputs/submission.csv
"""
import argparse
import os

from possession import dgate, growd, ingap, structure, subgap
from possession.baseline_plus import predict
from possession.data import load_match


def ingap_ph(match):
    """Ball-free P(H) per frame from a model trained on the external games only (never on the match)."""
    from possession import ballfree, external
    tabs = []
    for gid in dgate.training_games():
        _, X, Xs, y = ballfree.game_table(external.load_game(gid))
        k = y >= 0
        tabs.append((X[k], Xs[k], y[k]))
    model = ballfree.fit(tabs, "hgb")
    return ballfree.predict_frames(model, match, pitch=(structure.PITCH_L, structure.PITCH_W))


def build_labels():
    """Baseline+, then the missing-ball D gate, then (if enabled) the in-gap decoder; logs each footprint."""
    match = load_match()
    labels = predict(match)
    if not dgate.ENABLED:                 # ingap is only safe on D-gated labels (Q17): forced off with the gate
        return labels
    model = dgate.build_model()
    labels, dead, fp = dgate.apply(labels, match, model=model)
    print("dgate:", fp)
    gated = labels
    labels, fp = growd.apply(gated, match, dead=dead)
    if fp:
        print("growd:", fp)
    if ingap.ENABLED:
        grown = labels
        labels = ingap.apply(grown, match.B, match.half, ingap_ph(match), dead=dead)
        print("ingap:", ingap.footprint(grown, labels, match.B, match.half, dead))
    if subgap.ENABLED:
        pf = dgate.frame_pd(*model, match.P, match.V, match.T, match.half)
        labels, fp = subgap.apply(labels, gated, match, pf, fired=dead)
        print("subgap:", fp)
    return labels


def audio_status_line():
    """`audio: ok|reason` for the run log. Never raises: audio is optional evidence (S26)."""
    try:
        from possession import audio
        return audio.status_line()
    except Exception:  # noqa: BLE001 - the submission build must not depend on audio at all
        return "audio: audio_error"


def write_csv(labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs/submission.csv")
    args = parser.parse_args(argv)
    write_csv(build_labels(), args.out)
    print("wrote", args.out)
    print(audio_status_line())


if __name__ == "__main__":
    main()
