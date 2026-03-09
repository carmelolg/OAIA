#!/usr/bin/env python3
# Parse total line coverage percentage from a coverage.json file produced by
# pytest-cov (--cov-report=json). Prints a single integer percentage to stdout.
# Usage: python scripts/ci/parse_coverage.py [path/to/coverage.json]
# Example: python scripts/ci/parse_coverage.py coverage.json

import json
import sys

coverage_file = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "coverage.json"

try:
    with open(coverage_file) as f:
        data = json.load(f)
    pct = round(data["totals"]["percent_covered"])
    print(pct)
except FileNotFoundError:
    print(f"ERROR: Coverage file not found: {coverage_file}", file=sys.stderr)
    sys.exit(1)
except (KeyError, TypeError) as e:
    print(f"ERROR: Unexpected format in {coverage_file}: {e}", file=sys.stderr)
    sys.exit(1)
