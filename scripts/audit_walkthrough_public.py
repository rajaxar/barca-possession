"""Publication audit for the exported walkthrough page. Exit code 0 = publishable, 1 = stop.

The audit is deliberately independent of the exporter's own bookkeeping: it re-derives the public
figure set by calling ``build_walkthrough.build_notebook(numbers, "public")``, hashes every file
that set names, and then checks the finished HTML against it.

What it proves (each a named check, printed PASS/FAIL):

* **standalone** -- no ``<script>``, ``<link>``, iframe/object/embed, web font or ``@import``; no
  ``http(s)://``, ``ftp://`` or ``file://``; every ``src``/``href`` is a ``data:`` URI.
* **figure set** -- every public figure is embedded, byte-identical to the files the
  public profile names, and the embedded base64 decodes.
* **no real tracking** -- no embedded image is byte-identical to a local-only rendered figure, and
  no decoded payload carries a local-only figure name. The only GIF is ``fig_spellf1``, the
  synthetic centrepiece; every ``figs/public/*.png`` is pixel-identical to a freshly rendered neutral
  placeholder, so it cannot be a rendered clip that was dropped into the public folder.
* **no match detail** -- no local-only figure name outside the appendix, no frame number, no clock
  time, no match-derived path (``notebooks/figs``, ``outputs/``, ``data/``, ``research/``,
  ``scratch/``, absolute home path, ``.mp4``, ``submission.csv``), no work-card id outside the
  appendix. Published decision ids (``D-xxx``) and the traceability appendix are allowed.

    cd solution
    .venv/bin/python scripts/audit_walkthrough_public.py
    .venv/bin/python scripts/audit_walkthrough_public.py --json ../scratch/P04/audit.json
"""
import argparse
import base64
import collections
import glob
import html
import hashlib
import json
import os
import re
import sys
import tempfile

SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOLUTION_DIR)
sys.path.insert(0, os.path.join(SOLUTION_DIR, "scripts"))

import build_walkthrough as bw  # noqa: E402

FIGS = os.path.join(SOLUTION_DIR, "notebooks", "figs")
DEFAULT_HTML = os.path.join(SOLUTION_DIR, "notebooks", "possession_walkthrough_public.html")

FRAME_ALLOW = {"2027"}  # the competition year, the only bare >=4-digit number that may appear
PATH_MARKERS = ("notebooks/figs", "outputs/", "data/", "research/", "scratch/", "/Users/",
                ".mp4", "submission.csv")
CARD_ID = re.compile(r"(?<![\w-])([PQBE]\d{2,3})(?![\w-])")
FRAME_NUMBER = re.compile(r"(?<![\d,.])(\d{4,6})(?![\d,.])")
CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with open(path, "rb") as f:
        return sha(f.read())


def text_of(fragment):
    """Visible text: drop data URIs, style/script blocks and tags, then unescape."""
    frag = re.sub(r"data:image/[a-z+]+;base64,[A-Za-z0-9+/=]+", " ", fragment)
    frag = re.sub(r"<(style|script)\b.*?</\1>", " ", frag, flags=re.S | re.I)
    frag = re.sub(r"<[^>]+>", " ", frag)
    return re.sub(r"\s+", " ", html.unescape(frag)).strip()


def compare_pixels(a_path, b_path):
    from PIL import Image, ImageChops
    a = Image.open(a_path).convert("RGBA")
    b = Image.open(b_path).convert("RGBA")
    if a.size != b.size:
        return False
    return ImageChops.difference(a, b).getbbox() is None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--json", help="write a machine-readable report here")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.html):
        raise SystemExit("missing %s; run scripts/build_walkthrough_html.py" % args.html)
    doc = open(args.html, encoding="utf-8").read()
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print("%s %-18s %s" % ("PASS" if ok else "FAIL", name, detail))

    # ---------------------------------------------------------------- the public profile on disk
    numbers, _ = bw.load_numbers()
    nb, _ = bw.build_notebook(numbers, "public")
    cells = ["".join(c["source"]) for c in nb["cells"]]
    srcs = re.findall(r'src="([^"]+)"', "\n".join(cells))
    expected = collections.Counter(
        file_sha(os.path.join(SOLUTION_DIR, "notebooks", s)) for s in srcs)
    public_src_set = set(srcs)

    local_only = sorted(n for n, (_, lo, _) in bw.FIGURES.items() if lo)
    rendered = []
    for ext in ("*.gif", "*.png"):
        rendered += glob.glob(os.path.join(FIGS, ext))
    local_files = sorted(p for p in rendered
                         if os.path.relpath(p, os.path.join(SOLUTION_DIR, "notebooks")) not in public_src_set)
    local_hashes = {file_sha(p) for p in local_files}

    # ---------------------------------------------------------------- standalone / no network
    lowered = doc.lower()
    bad_tags = [t for t in ("<script", "<link", "<iframe", "<object", "<embed", "@import")
                if t in lowered]
    check("no-script", not bad_tags,
          "no JS, stylesheet link, iframe, embed or @import" if not bad_tags
          else "found %s" % ", ".join(bad_tags))

    urls = sorted(set(re.findall(r"(?:https?|ftp|file)://[^\s\"'<>]+", doc)))
    external_attrs = [m for m in re.findall(r'(?:src|href|poster)\s*=\s*"([^"]*)"', doc)
                      if not m.startswith("data:")]
    css_urls = [m for m in re.findall(r"url\(([^)]*)\)", doc) if not m.strip("'\"").startswith("data:")]
    check("no-network", not urls and not external_attrs and not css_urls,
          "all %d images inline, no URL, no non-data src/href/url()" % len(srcs)
          if not (urls or external_attrs or css_urls)
          else "urls=%s attrs=%s css=%s" % (urls[:3], external_attrs[:3], css_urls[:3]))

    # ---------------------------------------------------------------- images decode and match
    payloads = re.findall(r"data:(image/[a-z+]+);base64,([A-Za-z0-9+/=]+)", doc)
    decoded = []
    decode_errors = []
    for mime, b64 in payloads:
        try:
            decoded.append((mime, base64.b64decode(b64, validate=True)))
        except Exception as exc:  # noqa: BLE001 - report, never crash
            decode_errors.append(str(exc))
    embedded = collections.Counter(sha(data) for _, data in decoded)
    check("images-decode", not decode_errors and len(decoded) == len(srcs),
          "%d data URIs decoded (%d expected, 0 errors)" % (len(decoded), len(srcs))
          if not decode_errors else "decode errors: %s" % decode_errors[:3])

    check("figure-set", embedded == expected,
          "embedded bytes identical to the %d public figure files" % len(srcs)
          if embedded == expected else
          "mismatch: %d embedded vs %d expected (%d unexpected)"
          % (sum(embedded.values()), sum(expected.values()), len(set(embedded) - set(expected))))

    leaked = sorted(set(embedded) & local_hashes)
    check("no-tracking", not leaked,
          "0 of %d local-only figure hashes embedded" % len(local_hashes)
          if not leaked else "%d local-only figure(s) embedded: %s" % (len(leaked), leaked[:3]))

    name_hits = []
    for _, data in decoded:
        raw = data.decode("latin-1")
        name_hits += [n for n in local_only if n in raw]
    check("names-in-b64", not name_hits,
          "no local-only figure name inside any embedded payload"
          if not name_hits else "name(s) found: %s" % sorted(set(name_hits)))

    # placeholders must be placeholders, not rendered clips dropped into figs/public/
    tmp_root = os.path.join(SOLUTION_DIR, "..", "scratch", "P04")
    os.makedirs(tmp_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="p04_audit_", dir=tmp_root)
    old_public = bw.PUBLIC_FIGS
    try:
        bw.PUBLIC_FIGS = tmp
        for name in local_only:
            bw._placeholder(name, bw.FIGURES[name][2])
        placeholder_bad = []
        placeholder_n = 0
        for s in srcs:
            if not s.startswith("figs/public/"):
                continue
            placeholder_n += 1
            fresh = os.path.join(tmp, os.path.basename(s))
            if not compare_pixels(os.path.join(SOLUTION_DIR, "notebooks", s), fresh):
                placeholder_bad.append(os.path.basename(s))
    finally:
        bw.PUBLIC_FIGS = old_public
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    check("placeholders", not placeholder_bad and placeholder_n == len(local_only),
          "%d/%d public placeholders match a fresh neutral render" % (placeholder_n, len(local_only))
          if not placeholder_bad else "differs from a fresh placeholder: %s" % placeholder_bad)

    gif_hashes = [sha(data) for mime, data in decoded if mime == "image/gif"]
    centre = file_sha(os.path.join(FIGS, "fig_spellf1.gif"))
    check("only-synthetic-gif", gif_hashes == [centre],
          "1 GIF, fig_spellf1 (synthetic centrepiece); 0 rendered clips"
          if gif_hashes == [centre] else "GIF payloads: %d, unexpected set" % len(gif_hashes))

    # ---------------------------------------------------------------- no match detail in the text
    main_html = doc[doc.index("<main>"):]
    cut = main_html.find('<div class="appendix">')
    if cut < 0:
        body_text, appendix_text = text_of(main_html), ""
    else:
        end = main_html.find("</div>", cut)
        end = len(main_html) if end < 0 else end + len("</div>")
        appendix_html = main_html[cut:end]
        outside_html = main_html[:cut] + main_html[end:]
        body_text = text_of(outside_html)
        appendix_text = text_of(appendix_html)
    all_text = body_text + " " + appendix_text

    body_names = sorted({n for n in local_only if n in body_text})
    check("no-local-names", not body_names,
          "no local-only figure name outside the appendix"
          if not body_names else "found outside appendix: %s" % body_names)

    frame_hits = sorted(set(FRAME_NUMBER.findall(all_text)) - FRAME_ALLOW)
    check("no-frame-numbers", not frame_hits,
          "no bare frame-like integer (allowlist: %s)" % ", ".join(sorted(FRAME_ALLOW))
          if not frame_hits else "frame-like numbers: %s" % frame_hits[:8])

    clock_hits = sorted(set(CLOCK_TIME.findall(all_text)))
    check("no-clock", not clock_hits,
          "no clock time (mm:ss)" if not clock_hits else "clock-like: %s" % clock_hits[:5])

    path_hits = sorted({p for p in PATH_MARKERS if p in all_text})
    check("no-match-paths", not path_hits,
          "no match-derived path" if not path_hits else "path markers: %s" % path_hits)

    card_hits = sorted({m.group(0) for m in CARD_ID.finditer(body_text)})
    check("no-card-ids", not card_hits,
          "no work-card id outside the appendix (D-xxx citations kept)"
          if not card_hits else "card ids outside appendix: %s" % card_hits)

    # ---------------------------------------------------------------- report
    size = os.path.getsize(args.html)
    passed = all(c["ok"] for c in checks)
    print("")
    print("%s: %d/%d checks, %.1f KiB (%d bytes), %d images, %d local-only figures verified absent"
          % ("RESULT PASS" if passed else "RESULT FAIL", sum(c["ok"] for c in checks), len(checks),
             size / 1024.0, size, len(srcs), len(local_files)))
    print("       html: %s" % args.html)
    if args.json:
        report = {"html": os.path.abspath(args.html), "bytes": size, "pass": passed,
                  "images": len(srcs), "local_only_figures": local_only,
                  "checks": checks}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
        print("       json: %s" % args.json)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
