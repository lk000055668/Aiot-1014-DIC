"""Verify local preview assets and download links, without browser automation."""
from pathlib import Path
import re
from urllib.request import urlopen
import json

root=Path(__file__).resolve().parent
paths={"/", "/styles.css", "/app.js", "/analysis.js", "/selection.js", "/favicon.svg", "/data/project.json", "/api/health"}
for filename in ("index.html","app.js"):
    text=(root/"dist"/filename).read_text(encoding="utf-8")
    paths.update("/"+p for p in re.findall(r'downloads/[A-Za-z0-9_.-]+',text))
for path in sorted(paths):
    with urlopen("http://127.0.0.1:8000"+path,timeout=10) as response:
        assert response.status==200,path
        body=response.read()
        assert body,path
        if path.endswith("project.json"):
            payload=json.loads(body)
            assert len(payload["models"])==6
            assert len(payload["featureCounts"])==5
print(f"PASS: {len(paths)} pages, assets, data files and report downloads return HTTP 200.")
