"""Serve the existing website and interactive Python API on loopback."""
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import argparse
from api.index import ApiHandlerMixin

class LocalHandler(ApiHandlerMixin, SimpleHTTPRequestHandler):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "dist"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(LocalHandler, directory=str(directory)))
    print(f"Local: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
