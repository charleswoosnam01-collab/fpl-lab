"""Inline data.json into index.html so the app works as a single file."""
import json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1] / "site"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "preview.html"
html = (root / "index.html").read_text()
data = (root / "data.json").read_text().replace("</", "<\\/")
out.write_text(html.replace("/*FPL_DATA*/", f"window.FPL_DATA={data};", 1))
print("wrote", out)
