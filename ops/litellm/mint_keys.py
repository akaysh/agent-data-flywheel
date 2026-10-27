"""Mint and revoke budget-capped LiteLLM virtual keys for workshop attendees.

  python -m ops.litellm.mint_keys mint   --count 100 --out keys.csv
  python -m ops.litellm.mint_keys revoke --from keys.csv

Needs LITELLM_URL and LITELLM_MASTER_KEY. Never hand out the master key or raw cloud keys.
"""
import argparse
import csv
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS = ["workshop-agent", "workshop-fixed-agent", "workshop-judge"]
DEFAULT_ALIAS_PREFIX = "odsc-west-2026"


@dataclass(frozen=True)
class KeyPolicy:
    max_budget_usd: float
    rpm_limit: int
    duration: str
    models: list[str]

    def __post_init__(self):
        if self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive: an uncapped key is a liability")
        if not self.models:
            raise ValueError("models must list the models a key may call")


def mint_keys(base_url: str, master_key: str, count: int, policy: KeyPolicy,
              alias_prefix: str = DEFAULT_ALIAS_PREFIX) -> list[dict]:
    return [_post(base_url, master_key, "/key/generate", {
        "key_alias": f"{alias_prefix}-{i:03d}", "max_budget": policy.max_budget_usd,
        "rpm_limit": policy.rpm_limit, "duration": policy.duration, "models": policy.models})
        for i in range(1, count + 1)]


def revoke_keys(base_url: str, master_key: str, key_aliases: list[str]) -> None:
    _post(base_url, master_key, "/key/delete", {"key_aliases": key_aliases})


def write_key_sheet(keys: list[dict], path: str | Path) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key_alias", "key"])
        writer.writeheader()
        writer.writerows({"key_alias": k["key_alias"], "key": k["key"]} for k in keys)


def _post(base_url: str, master_key: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    mint = sub.add_parser("mint")
    mint.add_argument("--count", type=int, required=True)
    mint.add_argument("--budget", type=float, default=2.0, help="USD cap per key")
    mint.add_argument("--rpm", type=int, default=20)
    mint.add_argument("--duration", default="4h", help="key lifetime, e.g. 4h")
    mint.add_argument("--out", type=Path, required=True)
    revoke = sub.add_parser("revoke")
    revoke.add_argument("--from", dest="sheet", type=Path, required=True)
    args = ap.parse_args()

    url, master = os.environ["LITELLM_URL"], os.environ["LITELLM_MASTER_KEY"]
    if args.command == "mint":
        policy = KeyPolicy(args.budget, args.rpm, args.duration, DEFAULT_MODELS)
        keys = mint_keys(url, master, args.count, policy)
        write_key_sheet(keys, args.out)
        print(f"{len(keys)} keys -> {args.out} (budget ${args.budget}, {args.rpm} rpm, {args.duration})")
    else:
        aliases = [row["key_alias"] for row in csv.DictReader(args.sheet.open())]
        revoke_keys(url, master, aliases)
        print(f"revoked {len(aliases)} keys")


if __name__ == "__main__":
    main()
