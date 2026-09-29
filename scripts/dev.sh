#!/usr/bin/env bash
set -e

COMMAND="${1:-test}"

case "$COMMAND" in
    install)
        pip install -e .
        ;;
    test)
        pytest tests/
        ;;
    lint)
        echo "lint placeholder — no linter configured yet"
        ;;
    demo)
        echo "demo not yet implemented — see Prompt 17"
        ;;
    benchmark)
        echo "benchmark not yet implemented — see Prompt 18"
        ;;
    *)
        echo "Usage: $0 {install|test|lint|demo|benchmark}"
        exit 1
        ;;
esac
