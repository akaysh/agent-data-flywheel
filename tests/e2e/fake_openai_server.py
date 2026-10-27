"""Scripted OpenAI-compatible server for wiring checks (no real LLM needed).

Usage: python tests/e2e/fake_openai_server.py <port>
Replies, in order: router, get_charges, issue_refund, final answer, simulator <END>.
Matches the refund-under-limit task, so run Harbor with a single concurrent trial.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SCRIPT = [
    {"content": '{"agents": ["charges"]}'},
    {"tool": ("get_charges", {"ride_id": "R200"})},
    {"tool": ("issue_refund", {"ride_id": "R200", "charge_id": "C201", "amount": 12.0})},
    {"content": "I've refunded your $12 airport fee. It shows up in 3-5 business days."},
    {"content": "<END>"},
]
state = {"i": 0}


def completion(step: dict) -> dict:
    message = {"role": "assistant", "content": step.get("content")}
    if "tool" in step:
        name, args = step["tool"]
        message["tool_calls"] = [{"id": f"call_{state['i']}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
    finish_reason = "tool_calls" if "tool" in step else "stop"
    return {"id": f"chatcmpl-{state['i']}", "object": "chat.completion", "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        step = SCRIPT[min(state["i"], len(SCRIPT) - 1)]
        body = json.dumps(completion(step)).encode()
        state["i"] += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
