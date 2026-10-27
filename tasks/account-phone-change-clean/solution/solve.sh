#!/bin/bash
# Oracle: writes the expected tool calls so the verifier should score 1.0.
# If the oracle fails, the task is broken, not the agent.
cp /solution/expected_calls.jsonl /app/tool_calls.jsonl
touch /app/transcript.jsonl
