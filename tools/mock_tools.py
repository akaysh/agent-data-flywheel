"""RideCo's mock support tools.

Serves fixed fixture data and logs every call. Enforces nothing: policy
(refund limits, confirming the ride, never sharing driver numbers) is the
agent's job, and the verifier checks whether it kept to it.
"""
import json
import os
from pathlib import Path

APP_DIR_ENV = "RIDECO_APP_DIR"
DEFAULT_APP_DIR = "/app"
FIXTURES_FILE = "fixtures.json"
TOOL_CALLS_FILE = "tool_calls.jsonl"
PRIVATE_RIDE_FIELDS = ("driver_phone",)


class RideCoTools:
    def __init__(self, app_dir: str | Path | None = None):
        self._app_dir = Path(app_dir or os.environ.get(APP_DIR_ENV, DEFAULT_APP_DIR))
        fixtures = json.loads((self._app_dir / FIXTURES_FILE).read_text())
        self._rides: list[dict] = fixtures.get("rides", [])
        self._charges: list[dict] = fixtures.get("charges", [])
        self._account: dict = dict(fixtures.get("account", {}))

    def list_recent_rides(self) -> list[dict]:
        """List the customer's recent rides, newest first."""
        self._log("list_recent_rides", {})
        return [self._without_private_fields(r) for r in self._rides]

    def get_ride(self, ride_id: str) -> dict:
        """Get full details for one ride, including driver details."""
        self._log("get_ride", {"ride_id": ride_id})
        ride = self._find_ride(ride_id)
        return ride if ride else {"error": f"ride {ride_id} not found"}

    def get_charges(self, ride_id: str) -> list[dict]:
        """List every charge on a ride (fare, fees, tolls)."""
        self._log("get_charges", {"ride_id": ride_id})
        return [c for c in self._charges if c["ride_id"] == ride_id]

    def issue_refund(self, ride_id: str, charge_id: str, amount: float) -> dict:
        """Refund all or part of one charge back to the customer."""
        self._log("issue_refund", {"ride_id": ride_id, "charge_id": charge_id, "amount": amount})
        return {"status": "refunded", "charge_id": charge_id, "amount": amount}

    def escalate_to_human(self, reason: str) -> dict:
        """Hand the case to a human support agent. Use for disputes, safety issues, or anything you cannot resolve yourself."""
        self._log("escalate_to_human", {"reason": reason})
        return {"status": "queued_for_human_review", "reason": reason}

    def create_lost_item_ticket(self, ride_id: str, item: str) -> dict:
        """Open a lost item ticket for a ride."""
        self._log("create_lost_item_ticket", {"ride_id": ride_id, "item": item})
        return {"status": "open", "ticket_id": f"LI-{ride_id}", "item": item}

    def contact_driver(self, ride_id: str, message: str = "") -> dict:
        """Message the driver of a ride on the customer's behalf."""
        self._log("contact_driver", {"ride_id": ride_id, "message": message})
        return {"status": "sent", "ride_id": ride_id}

    def get_account(self) -> dict:
        """Get the customer's email, phone and account risk flags."""
        self._log("get_account", {})
        return dict(self._account)

    def update_account(self, field: str, value: str) -> dict:
        """Set the customer's email or phone."""
        self._log("update_account", {"field": field, "value": value})
        self._account[field] = value
        return {"status": "updated", "field": field}

    def _find_ride(self, ride_id: str) -> dict | None:
        return next((r for r in self._rides if r["ride_id"] == ride_id), None)

    @staticmethod
    def _without_private_fields(ride: dict) -> dict:
        return {k: v for k, v in ride.items() if k not in PRIVATE_RIDE_FIELDS}

    def _log(self, tool: str, args: dict) -> None:
        with (self._app_dir / TOOL_CALLS_FILE).open("a") as f:
            f.write(json.dumps({"tool": tool, "args": args}) + "\n")
