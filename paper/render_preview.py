#!/usr/bin/env python
"""Compile-and-preview for the paper: renders main.pdf page by page (PyMuPDF) into one self-contained HTML
that is published as the paper artifact after EVERY edit (user 2026-09-04: "每次修改都要渲染").

    cd paper && make && python render_preview.py [--zoom 1.5] [--out <html>]

Prints the HTML path and page count; publish it with the Artifact tool to the same URL each time.
"""
import argparse, base64, datetime, pathlib, subprocess
import fitz  # PyMuPDF

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", default="main.pdf")
ap.add_argument("--zoom", type=float, default=1.5, help="1.0 = 72 dpi; 1.5 ~ 108 dpi")
ap.add_argument("--out", default="/tmp/claude-3565/-iridisfs-scratch-ts1v23-workspace-noKslot-clean/3b618710-38c6-4bfc-a19a-8bb6fa615348/scratchpad/paper_pdf.html")
a = ap.parse_args()
doc = fitz.open(a.pdf)
pngs = [page.get_pixmap(matrix=fitz.Matrix(a.zoom, a.zoom)).tobytes("png") for page in doc]
commit = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True, cwd="..").stdout.strip()
dirty = subprocess.run(["git", "status", "--short", "--", "."], capture_output=True, text=True, cwd=".").stdout.strip()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
css = """
:root{--bg:#f3f4f2;--ink:#1c2128;--mute:#66717d;--rule:#d5dad6;--card:#ffffff;--accent:#3b5b7a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15181c;--ink:#e6e8e6;--mute:#98a2ad;--rule:#2b3138;--card:#1d2126;--accent:#8fb3d6}}
:root[data-theme="dark"]{--bg:#15181c;--ink:#e6e8e6;--mute:#98a2ad;--rule:#2b3138;--card:#1d2126;--accent:#8fb3d6}
body{background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif;margin:0}
main{max-width:980px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 6px;text-wrap:balance} .meta{color:var(--mute);font-size:13px;margin-bottom:14px}
.meta code{font-family:"IBM Plex Mono",monospace;font-size:12px}
nav{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px} nav a{color:var(--accent);text-decoration:none;font-size:12px;border:1px solid var(--rule);border-radius:4px;padding:2px 8px}
.page{background:#fff;border:1px solid var(--rule);margin:0 0 18px} .page img{display:block;width:100%;height:auto}
.pnum{color:var(--mute);font-size:12px;margin:0 0 4px} .note{color:var(--mute);font-size:12px;margin-top:16px}
"""
h = [f"<title>TopX 论文稿</title>\n<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap\">\n<style>{css}</style>\n<main>",
     "<h1>One Model, Many Skeletons — 论文稿（ICLR 2027 模板）</h1>",
     f"<p class=\"meta\">编译于 {now} · {len(pngs)} 页 · 源 paper/main.tex + sections/ · 仓库 HEAD <code>{commit}</code>"
     f"{' · paper/ 有未提交改动' if dirty else ''} · PDF：paper/main.pdf。每次改稿都会重新编译并重发到本链接。</p>",
     "<nav>" + "".join(f"<a href=\"#p{i+1}\">第 {i+1} 页</a>" for i in range(len(pngs))) + "</nav>"]
for i, png in enumerate(pngs):
    h.append(f"<p class=\"pnum\" id=\"p{i+1}\">第 {i+1} / {len(pngs)} 页</p><div class=\"page\"><img src=\"data:image/png;base64,{base64.b64encode(png).decode()}\" alt=\"page {i+1}\"></div>")
h.append("<p class=\"note\">状态：Abstract、Introduction、Related Work、Method 正文已写；Experiments 与图 1 待写；\\todo 处为待定数字或来源。</p></main>")
out = pathlib.Path(a.out); out.write_text("\n".join(h))
print(f"{out} {out.stat().st_size/1e6:.1f} MB, {len(pngs)} pages")
