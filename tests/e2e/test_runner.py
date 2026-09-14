#!/usr/bin/env python3
"""Modular3 E2E Test Suite Discovery and Execution Runner.

Standalone CLI test runner supporting tier filtering, formatted summary reporting (text/json),
and clean exit codes (0 on success, 1 on failure).
"""

import argparse
import json
import os
import sys
import time
import unittest
from pathlib import Path

try:
    import numpy
except ImportError:
    project_python = ""
    if os.path.exists(project_python) and sys.executable != project_python:
        os.execv(project_python, [project_python] + sys.argv)



class E2ETestResult(unittest.TestResult):
    """Custom TestResult collecting detailed execution and outcome metrics."""

    def __init__(self, stream=None, descriptions=None, verbosity=1):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []
        self.start_time = None
        self.stop_time = None

    def startTest(self, test):
        super().startTest(test)
        if self.start_time is None:
            self.start_time = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)

    def stopTestRun(self):
        super().stopTestRun()
        self.stop_time = time.time()

    @property
    def duration(self):
        if self.start_time and self.stop_time:
            return self.stop_time - self.start_time
        return 0.0

    @property
    def passed_count(self):
        return len(self.successes)

    @property
    def skipped_count(self):
        return len(self.skipped)

    @property
    def failure_count(self):
        return len(self.failures)

    @property
    def error_count(self):
        return len(self.errors)

    @property
    def total_executed(self):
        return self.testsRun


def get_tier_directory(tier: str, base_dir: Path) -> Path:
    """Resolve directory path for requested test tier."""
    tier_map = {
        "1": base_dir / "tier1_features",
        "2": base_dir / "tier2_boundaries",
        "3": base_dir / "tier3_pairwise",
        "4": base_dir / "tier4_scenarios",
    }
    return tier_map.get(str(tier))


def build_test_suite(selected_tiers: list, base_dir: Path) -> unittest.TestSuite:
    """Discover tests from selected tiers and compile into a single TestSuite."""
    loader = unittest.TestLoader()
    combined_suite = unittest.TestSuite()

    tier_keys = ["1", "2", "3", "4"] if "all" in selected_tiers else selected_tiers

    for tier in sorted(tier_keys):
        tier_dir = get_tier_directory(tier, base_dir)
        if tier_dir and tier_dir.exists():
            discovered = loader.discover(
                start_dir=str(tier_dir),
                pattern="test_*.py",
                top_level_dir=str(base_dir.parent.parent),
            )
            combined_suite.addTest(discovered)

    return combined_suite


def run_e2e_suite(tiers: list, verbosity: int = 1, failfast: bool = False):
    """Execute tests for designated tiers and return (result, test_suite)."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent

    # Ensure workspace root and test directory are on python path
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    suite = build_test_suite(tiers, script_dir)
    result = E2ETestResult(verbosity=verbosity)
    result.failfast = failfast

    suite.run(result)
    return result, suite


def generate_json_summary(result: E2ETestResult, tiers: list) -> dict:
    """Generate structured JSON report payload."""
    failures_summary = [
        {"test": str(t), "err": err} for t, err in result.failures
    ]
    errors_summary = [
        {"test": str(t), "err": err} for t, err in result.errors
    ]
    skipped_summary = [
        {"test": str(t), "reason": reason} for t, reason in result.skipped
    ]

    return {
        "suite": "Modular3 E2E Test Suite",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tiers": tiers,
        "summary": {
            "total_executed": result.total_executed,
            "passed": result.passed_count,
            "skipped": result.skipped_count,
            "failures": result.failure_count,
            "errors": result.error_count,
            "success": result.wasSuccessful(),
            "duration_seconds": round(result.duration, 4),
        },
        "failures": failures_summary,
        "errors": errors_summary,
        "skipped": skipped_summary,
    }


def print_text_summary(result: E2ETestResult, tiers: list):
    """Print human-readable test summary report."""
    print("\n" + "=" * 70)
    print(f"MODULAR3 E2E TEST SUITE RUNNER — TIERS: {', '.join(tiers)}")
    print("=" * 70)
    print(f"Total Tests Executed : {result.total_executed}")
    print(f"Passed               : {result.passed_count}")
    print(f"Skipped (pending src): {result.skipped_count}")
    print(f"Failures             : {result.failure_count}")
    print(f"Errors               : {result.error_count}")
    print(f"Total Execution Time : {result.duration:.4f}s")
    print("-" * 70)

    if result.failures:
        print("\n--- FAILURES ---")
        for test, trace in result.failures:
            print(f"\nFAIL: {test}\n{trace}")

    if result.errors:
        print("\n--- ERRORS ---")
        for test, trace in result.errors:
            print(f"\nERROR: {test}\n{trace}")

    if result.wasSuccessful():
        print("OVERALL STATUS: SUCCESS (0 failures, 0 errors)")
    else:
        print("OVERALL STATUS: FAILED")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Modular3 Emergent Order Simulator E2E Test Runner"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        help="Test tier to execute: '1', '2', '3', '4', or 'all' (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Summary report output format (default: text)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to write JSON or text summary output to",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Verbosity level",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure or error",
    )

    args = parser.parse_args()

    # Parse tier argument
    tier_arg = args.tier.lower().strip()
    if tier_arg == "all":
        tiers = ["all"]
    elif "," in tier_arg:
        tiers = [t.strip() for t in tier_arg.split(",") if t.strip() in ["1", "2", "3", "4"]]
    elif tier_arg in ["1", "2", "3", "4"]:
        tiers = [tier_arg]
    else:
        print(f"Invalid tier argument: '{args.tier}'. Choose from '1', '2', '3', '4', 'all'.", file=sys.stderr)
        sys.exit(2)

    result, suite = run_e2e_suite(tiers=tiers, verbosity=args.verbose, failfast=args.failfast)

    if args.format == "json":
        summary_data = generate_json_summary(result, tiers)
        output_str = json.dumps(summary_data, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str + "\n")
        else:
            print(output_str)
    else:
        print_text_summary(result, tiers)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json.dumps(generate_json_summary(result, tiers), indent=2) + "\n")

    # Exit code: 0 if successful (passed or skipped, 0 failures/errors), 1 on failure
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
