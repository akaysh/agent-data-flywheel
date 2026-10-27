import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ops.litellm.mint_keys import KeyPolicy, mint_keys, revoke_keys, write_key_sheet


class StubProxy(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubProxy.requests.append((self.path, self.headers.get("Authorization"), body))
        reply = {"key": f"sk-{len(StubProxy.requests)}", "key_alias": body.get("key_alias")}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def proxy_url():
    StubProxy.requests = []
    server = HTTPServer(("127.0.0.1", 0), StubProxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


POLICY = KeyPolicy(max_budget_usd=2.0, rpm_limit=20, duration="3h",
                   models=["workshop-agent", "workshop-judge"])


def test_mint_keys_sends_budget_rate_limit_expiry_and_model_allowlist(proxy_url):
    keys = mint_keys(proxy_url, "master-secret", 2, POLICY, alias_prefix="odsc")
    assert [k["key_alias"] for k in keys] == ["odsc-001", "odsc-002"]
    path, auth, body = StubProxy.requests[0]
    assert path == "/key/generate" and auth == "Bearer master-secret"
    assert body["max_budget"] == 2.0 and body["rpm_limit"] == 20 and body["duration"] == "3h"
    assert body["models"] == ["workshop-agent", "workshop-judge"]


def test_revoke_keys_deletes_by_alias(proxy_url):
    revoke_keys(proxy_url, "master-secret", ["odsc-001", "odsc-002"])
    path, _, body = StubProxy.requests[0]
    assert path == "/key/delete" and body == {"key_aliases": ["odsc-001", "odsc-002"]}


def test_key_sheet_lists_alias_and_key_for_handing_out(tmp_path):
    path = tmp_path / "keys.csv"
    write_key_sheet([{"key_alias": "odsc-001", "key": "sk-abc"}], path)
    rows = list(csv.DictReader(path.open()))
    assert rows == [{"key_alias": "odsc-001", "key": "sk-abc"}]


def test_refuses_zero_budget_or_unbounded_keys():
    with pytest.raises(ValueError, match="budget"):
        KeyPolicy(max_budget_usd=0, rpm_limit=20, duration="3h", models=["m"])
    with pytest.raises(ValueError, match="models"):
        KeyPolicy(max_budget_usd=1, rpm_limit=20, duration="3h", models=[])
