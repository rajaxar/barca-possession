"""Render either walkthrough profile to one standalone HTML file.

This is an export of the notebook the builder already produces, not a second source: the cells,
prose, captions and figures all come from ``build_walkthrough.build_notebook(numbers, profile)``.
Every image is inlined as a ``data:`` URI, there is no JavaScript, no web font and no external
request, so the page opens offline and can be read at 700-900 px.

    cd solution
    .venv/bin/python scripts/build_walkthrough_html.py                      # public (default)
    .venv/bin/python scripts/build_walkthrough_html.py --profile local      # local reading copy
    .venv/bin/python scripts/audit_walkthrough_public.py                    # publication audit, exit 0/1

The public profile replaces every real-tracking figure with a neutral placeholder (D-002/D-026), so
the page is safe to hand to a reader. It is still **not published** here: the file stays on disk
until the human decides. The local profile inlines the rendered clips; it is **personal and
local-only**, must never be committed, published or uploaded, and its footer records the CSV sha and
the git commit it was built from.

One presentation-only edit is made in this exporter, never in the notebook: internal work-card ids
in prose (``(E83)``) are removed, because a reader has no use for them and the card requires no card
id outside the appendix. Published decision ids (``D-xxx``) and the traceability appendix are kept.
"""
import argparse
import base64
import hashlib
import html
import os
import re
import subprocess
import sys

SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOLUTION_DIR)
sys.path.insert(0, os.path.join(SOLUTION_DIR, "scripts"))

import build_walkthrough as bw  # noqa: E402  (same folder; no pipeline import)

DEFAULT_OUT = {
    "public": os.path.join(SOLUTION_DIR, "notebooks", "possession_walkthrough_public.html"),
    "local": os.path.join(SOLUTION_DIR, "notebooks", "possession_walkthrough_local.html"),
}
MIME = {".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# Internal work-card ids have no meaning to a reader. Decision ids (D-xxx) are published citations
# and stay. Only parenthesised ids occur in the prose today; the audit checks for leftovers.
_INTERNAL_ID = re.compile(r"\s*\(\s*[PQBE]\d{2,3}\s*\)")

CSS = """
:root {
  --ink: #1c1f21; --muted: #5a6368; --rule: #dfe3e2; --panel: #f3f6f4;
  --accent: #1a5fb4; --accent-soft: #e8f0f8;
}
* { box-sizing: border-box; }
html { background: #f7f8f7; }
body {
  margin: 0; color: var(--ink); background: #f7f8f7;
  font: 17px/1.66 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 840px; margin: 0 auto; padding: 56px 30px 100px; background: #fff; }
h1 { font-size: 2.15rem; line-height: 1.14; letter-spacing: -0.02em; margin: 0 0 0.5em; }
h2 {
  font-size: 1.48rem; line-height: 1.25; margin: 2.4em 0 0.7em; padding-top: 0.7em;
  border-top: 1px solid var(--rule);
}
p { margin: 0 0 1.05em; }
strong { font-weight: 650; }
em { font-style: italic; }
code {
  font: 0.86em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: #f0f3f1; padding: 0.1em 0.32em; border-radius: 3px;
}
figure { margin: 0 0 1.7em; }
figure img { display: block; width: 100%; height: auto; margin: 0 auto; border-radius: 2px; }
figcaption { font-size: 0.86rem; line-height: 1.5; color: var(--muted); margin-top: 0.45em; }
.figpair { display: flex; gap: 14px; flex-wrap: wrap; }
.figpair > div { flex: 1 1 300px; min-width: 0; padding: 0; }
.figpair figure { margin-bottom: 1.2em; }
table {
  width: 100%; border-collapse: collapse; margin: 1.3em 0 1.7em;
  font-size: 0.84rem; line-height: 1.45;
}
th, td { text-align: left; vertical-align: top; padding: 0.48em 0.55em; border-bottom: 1px solid var(--rule); }
th { background: var(--panel); font-weight: 650; }
tbody tr:nth-child(even) { background: #fafbfa; }
.appendix { margin-top: 1em; }
.appendix table { font-size: 0.78rem; }
.appendix code { background: none; padding: 0; }
@media (max-width: 640px) {
  body { font-size: 16px; }
  main { padding: 32px 16px 64px; }
  h1 { font-size: 1.75rem; }
  h2 { font-size: 1.3rem; }
  table { font-size: 0.78rem; }
  th, td { padding: 0.4em 0.4em; }
}
@media print {
  html, body { background: #fff; }
  main { max-width: none; padding: 0; }
  h2 { break-after: avoid; page-break-after: avoid; }
  figure, table { break-inside: avoid; page-break-inside: avoid; }
}
"""

# Only the local page carries a build footer. Keeping it out of the public CSS is what makes the
# generalized exporter byte-identical on the public path when the notebook is unchanged.
FOOTER_CSS = """
footer.build {
  margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--rule);
  font-size: 0.78rem; line-height: 1.55; color: var(--muted);
}
footer.build code { font-size: 0.95em; word-break: break-all; }
"""


# --------------------------------------------------------------------------- inline helpers

def _esc(text):
    return html.escape(text, quote=False)


def _inline(text):
    """Escape then render code spans, bold and italic (the only inline markdown in the prose)."""
    codes = []

    def keep(m):
        codes.append(_esc(m.group(1)))
        return "\x00%d\x00" % (len(codes) - 1)

    out = re.sub(r"`([^`]+)`", keep, _esc(text))
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", out)
    out = re.sub(r"\x00(\d+)\x00", lambda m: "<code>%s</code>" % codes[int(m.group(1))], out)
    return out


def _table(rows):
    """rows: list of raw pipe-table lines (header, delimiter, body)."""
    def cells(line):
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [c.strip() for c in line.split("|")]

    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    out = ["<table>", "<thead><tr>"]
    out += ["<th>%s</th>" % _inline(c) for c in head]
    out += ["</tr></thead>", "<tbody>"]
    for row in body:
        out.append("<tr>" + "".join("<td>%s</td>" % _inline(c) for c in row) + "</tr>")
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def _markdown(text):
    text = _INTERNAL_ID.sub("", text)
    lines = text.splitlines()
    out, para = [], []

    def flush():
        if para:
            out.append("<p>%s</p>" % _inline(" ".join(para)))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            level = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (level, _inline(m.group(2)), level))
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_table(block))
            continue
        para.append(stripped)
        i += 1
    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------- images / cells

def _data_uri(src, profile="public"):
    path = os.path.join(SOLUTION_DIR, "notebooks", src)
    if not os.path.isfile(path):
        raise SystemExit("missing figure for the %s profile: %s" % (profile, path))
    ext = os.path.splitext(path)[1].lower()
    mime = MIME.get(ext)
    if mime is None:
        raise SystemExit("unsupported image type: %s" % path)
    with open(path, "rb") as f:
        payload = f.read()
    return "data:%s;base64,%s" % (mime, base64.b64encode(payload).decode("ascii")), len(payload)


def _html_cell(source, profile="public"):
    """Inline the images, turn (img, caption) pairs into <figure>, make the two-up block wrap."""
    source = source.replace('style="display:flex;"', 'class="figpair"')

    def inline_img(m):
        uri, _ = _data_uri(m.group(1), profile)
        return 'src="%s"' % uri

    source = re.sub(r'src="([^"]+)"', inline_img, source)

    def to_figure(m):
        img, caption = m.group(1), m.group(2)
        alt = re.sub(r"<[^>]+>", "", caption).strip()
        img = img.replace("<img ", '<img alt="%s" ' % html.escape(alt, quote=True), 1)
        return "<figure>%s<figcaption>%s</figcaption></figure>" % (img, caption.strip())

    source = re.sub(
        r'(<img\b[^>]*>)\s*<div class="caption"[^>]*>(.*?)</div>',
        to_figure, source, flags=re.S)
    return source


def _footer(numbers):
    """Deterministic build footer: no wall-clock time, so rebuilds are comparable."""
    csv_path = os.path.join(SOLUTION_DIR, "outputs", "submission.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, "rb") as f:
            csv_sha = hashlib.sha256(f.read()).hexdigest()
    else:
        csv_sha = "unavailable (outputs/submission.csv is missing)"
    try:
        commit = subprocess.check_output(
            ["git", "-C", SOLUTION_DIR, "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown (not a git checkout)"
    return (
        '<footer class="build">'
        'Local reading copy, built from the frozen CSV at sha256 <code>%s</code> '
        'and git commit <code>%s</code>. '
        'It contains real-tracking figures rendered from match and external-game data: '
        '<strong>do not commit, publish or upload it</strong> (D-002/D-026).'
        '</footer>' % (html.escape(csv_sha), html.escape(commit))
    )


def render_html(numbers, profile="public"):
    nb, _ = bw.build_notebook(numbers, profile)
    body, images = [], []
    in_appendix = False
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if source.lstrip().startswith("## Appendix"):
            in_appendix = True
            body.append('<div class="appendix">')
        if "<img" in source:
            images.extend(re.findall(r'src="([^"]+)"', source))
            body.append(_html_cell(source, profile))
        else:
            body.append(_markdown(source))
    if in_appendix:
        body.append("</div>")
    if profile == "local":
        body.append(_footer(numbers))
    doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reading the ball — a traceable possession walkthrough</title>
<meta name="description" content="A traceable walkthrough of a frame-level football possession model, More than a Hack 2027.">
<meta name="generator" content="solution/scripts/build_walkthrough_html.py (%s profile)">
<style>%s</style>
</head>
<body>
<main>
%s
</main>
</body>
</html>
""" % (profile, CSS + (FOOTER_CSS if profile == "local" else ""), "\n\n".join(body))
    return doc, images


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=("public", "local"), default="public")
    ap.add_argument("--out", help="default: notebooks/possession_walkthrough_<profile>.html")
    args = ap.parse_args(argv)
    out = args.out or DEFAULT_OUT[args.profile]
    numbers, _ = bw.load_numbers()
    doc, images = render_html(numbers, args.profile)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    size = os.path.getsize(out)
    print("wrote %s (%s profile)" % (out, args.profile))
    print("  %.1f KiB, %d inlined images (%d distinct files)" % (
        size / 1024.0, len(images), len(set(images))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
