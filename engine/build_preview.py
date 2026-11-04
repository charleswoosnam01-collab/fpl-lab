"""Inline data.json (and the solver) into index.html so the app works as a single file."""
import base64, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1] / "site"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "preview.html"
html = (root / "index.html").read_text()
data = (root / "data.json").read_text().replace("</", "<\\/")
inject = f"window.FPL_DATA={data};"
vendor = root / "vendor"
if (vendor / "highs.js").exists() and "--no-solver" not in sys.argv:
    js = (vendor / "highs.js").read_text().replace("</script", "<\\/script")
    wasm = base64.b64encode((vendor / "highs.wasm").read_bytes()).decode()
    html = html.replace("<script>\n/*FPL_DATA*/", f"<script>{js}\nwindow.HighsFactory = Module;</script>\n<script>window.HIGHS_WASM_B64=\"{wasm}\";</script>\n<script>\n/*FPL_DATA*/", 1)
out.write_text(html.replace("/*FPL_DATA*/", inject, 1))
print("wrote", out, round(out.stat().st_size / 1e6, 1), "MB")
