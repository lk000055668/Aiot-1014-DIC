"""The same JSON endpoints are used locally and by a Vercel Python function."""
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import json
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regression_workflow import (
    AnalysisError, preview_dataset, analyze, predict, export_pipeline, MAX_CSV_BYTES
)
from selection_workflow import analyze_selection, predict_selection, descriptive_selection
from threadpoolctl import threadpool_limits


class ApiHandlerMixin:
    def operation(self):
        parsed = urlsplit(self.path)
        return parse_qs(parsed.query).get("operation", [parsed.path.rsplit("/", 1)[-1]])[0]

    def reply(self, status, body, content_type="application/json", filename=None):
        if content_type == "application/json":
            body = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # The browser may cancel a request after switching datasets.

    def do_GET(self):
        if self.path.startswith("/api"):
            if self.operation() == "health":
                self.reply(200, {"status": "ok", "mode": "interactive-regression"})
            else:
                self.reply(404, {"error": "Unknown API endpoint."})
        elif isinstance(self, SimpleHTTPRequestHandler):
            SimpleHTTPRequestHandler.do_GET(self)
        else:
            self.reply(404, {"error": "Not found."})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length < 1 or length > MAX_CSV_BYTES * 3:
                raise AnalysisError("Request is empty or too large; CSV uploads are limited to 1 MB.")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise AnalysisError("Send the request as application/json.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise AnalysisError("Request must be a JSON object.")
            operation = self.operation()
            with threadpool_limits(limits=1):
                if operation == "dataset":
                    response = preview_dataset(payload)
                elif operation == "analyze":
                    response = analyze(payload)
                elif operation == "selection":
                    response = analyze_selection(payload)
                elif operation == "selection-descriptive":
                    response = descriptive_selection(payload)
                elif operation == "selection-predict":
                    response = predict_selection(payload)
                elif operation == "predict":
                    response = predict(payload)
                elif operation == "model":
                    content, filename = export_pipeline(payload)
                    self.reply(200, content, "application/octet-stream", filename)
                    return
                else:
                    self.reply(404, {"error": "Unknown API endpoint."})
                    return
            self.reply(200, response)
        except (AnalysisError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.reply(400, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            self.reply(500, {"error": "Analysis failed unexpectedly. Check the dataset and retry."})


class handler(ApiHandlerMixin, BaseHTTPRequestHandler):
    """Vercel's file-based Python entrypoint."""
