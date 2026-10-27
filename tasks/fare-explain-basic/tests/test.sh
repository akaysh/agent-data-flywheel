#!/bin/bash
# Deterministic checks only. Judge scoring runs separately (module 2).
mkdir -p /logs/verifier
python3 /tests/check_state.py /tests/checks.json /app /logs/verifier
