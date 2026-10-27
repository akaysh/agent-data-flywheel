# Stage runbook

Everything here was run end to end against the committed cache. Commands run from the repo root.

## Before you go on stage
- [ ] Docker is running (`docker info`) and `harbor --version` works.
- [ ] `pytest -q` passes (no network needed).
- [ ] `harbor run -p tasks -a oracle -n 5` scores 1.0 on all tasks (the tasks themselves are sound).
- [ ] `git status` shows no `scenarios/regressions.yaml` and no `tasks/regression-*` (attendees start without them).
- [ ] Browser tabs: the Colab notebook, `harbor view` if you want trial pages.
- [ ] Plan B ready: the notebook with `OFFLINE = True` needs nothing from the network.

## 5 to 17 min: the harness (on stage, Docker)
```bash
OFFLINE=1 harbor run -p tasks \
  --agent-import-path harbor_adapter.rideco_agent:RideCoAgent -n 5
```
Takes about two minutes. Expect 10 of 15 passing. The five failures are the planted bugs:
`account-email-and-lost-wallet`, `account-takeover-signal`, `driver-phone-request`,
`lost-phone-cleaning-fee`, `refund-over-limit`.
Run it before you start talking about it and narrate while it runs. Show one trial's
`verifier/checks.json` for a failing task.

## 17 to 32 min: the judge (notebook)
Section 3 of the notebook. The beat to land: `escalation-claimed-as-refund` has verifier reward 1 and
the judge fails `honest_about_actions`. Then show the vague rubric next to the good one.
Say plainly that the expert labels in the repo were produced by a strong model, not by a human team, and what the kappas are.

## 32 to 47 min: traces to training data (notebook)
Sections 4 and 5. Show `scenarios/regressions.yaml`, then one preference pair.

## 47 to 55 min: close the loop (on stage, Docker)
First attempt, prompts only, same small model:
```bash
OFFLINE=1 PROMPT_VERSION=v2 harbor run -p tasks \
  --agent-import-path harbor_adapter.rideco_agent:RideCoAgent -n 5
```
Expect 10 of 15 again, but different tasks fail: `account-email-and-lost-wallet`,
`account-takeover-signal`, `refund-over-limit`, `refund-two-charges-over-total-limit`,
`refund-which-ride`. Open one trial's `agent/rollout.log` or the notebook transcript: the small
model says "I've sent this to our team" and never calls `escalate_to_human`. A prompt can't fix that.

Second attempt, prompts plus a stronger model:
```bash
OFFLINE=1 PROMPT_VERSION=v2 AGENT_ROLE=fixed_agent harbor run -p tasks \
  --agent-import-path harbor_adapter.rideco_agent:RideCoAgent -n 5
```
Expect 14 of 15 passing. `safety-complaint-escalate` still fails: the stronger model refunds and then escalates.
That is a good place to say "the fix didn't catch everything, and now we know which task to look at."

## If something breaks
| Symptom | Do this |
| --- | --- |
| Docker won't start | Run the notebook only. Sections 2 to 5 cover the same ground with the Colab runner. |
| `CacheMissError` | You changed a prompt, tool, task or rubric. Re-record (README, "Offline mode") or revert. |
| Wifi dies | Nothing changes. Everything runs from `cache/` with `OFFLINE=1`. |
| Harbor trial hangs | Ctrl-C, rerun with `-n 1` on the failing task. Container logs are in the job directory. |

## Re-recording (needs a live model)
See README, "Offline mode and the cache". Use `.env` for the endpoint and keys. Never commit `.env`.
