"""
시연 화면 서버 (추가 패키지 없이 파이썬 기본 기능만 사용)

  python src/server.py          → http://localhost:8600

- GET  /            web/index.html
- GET  /api/task    과제 문구
- POST /api/grade   {"essay": "..."} → {"score": 4, "rationale": "..."}
채점은 최종 채점 기준(prompts/criteria_final.md)과 Claude Haiku 5.5로 한다. .env 에 ANTHROPIC_API_KEY 필요.
"""

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asap import load, task_info  # noqa: E402
from loop import TOPIC, grade_one  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PORT = 8600

ASSIGNMENT, SOURCES = task_info(load(), TOPIC)
CRITERIA = (ROOT / "prompts" / "criteria_final.md").read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/task":
            return self._send(200, {"assignment": ASSIGNMENT})
        rel = "index.html" if self.path in ("/", "") else self.path.lstrip("/").split("?")[0]
        f = (WEB / rel).resolve()
        if WEB.resolve() not in f.parents and f != WEB.resolve() / "index.html" or not f.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".woff2":
            ctype = "font/woff2"
        self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/grade":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            essay = (body.get("essay") or "").strip()
        except json.JSONDecodeError:
            return self._send(400, {"error": "요청 형식이 올바르지 않습니다."})
        if len(essay.split()) < 20:
            return self._send(400, {"error": "에세이가 너무 짧습니다. 20단어 이상 입력하세요."})
        g = grade_one(CRITERIA, ASSIGNMENT, SOURCES, {"id": "web", "text": essay}, "web")
        if g["error"]:
            return self._send(502, {"error": "채점 결과를 읽지 못했습니다. 다시 채점해 주세요."})
        self._send(200, {"score": g["ai"], "rationale": g["rationale"]})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
