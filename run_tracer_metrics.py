#!/usr/bin/env python3
# Copyright 2024 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =====================================================================
"""Collects tracer performance metrics for StableHLO files.

This script walks through the ``modelgarden/nlp/stablehlo`` directory,
invokes ``tracer_perf_main`` for each ``.stablehlo`` file and stores the
extracted JSON metrics and command logs in a mirrored directory tree
under ``modelgarden/nlp/metrics``.

The script can also be run in *synthetic bug* mode using ``--synthetic-bugs``.
When enabled, flags are passed to ``tracer_perf_main`` to activate
synthetic bugs used for testing the concurrency tracer.  Individual
synthetic bugs can be toggled with ``--bug-wait-for-streams``,
``--bug-collective-done`` and ``--bug-remove-control-deps``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import logging
from pathlib import Path

# Configure logging
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt=datefmt,
)
logger = logging.getLogger(__name__)

STABLEHLO_ROOT = Path("modelgarden/nlp/stablehlo")
METRICS_ROOT = Path("modelgarden/nlp/metrics")
BUG_METRICS_ROOT = METRICS_ROOT / "../synthetic"


def extract_json(output: str) -> dict:
    start_marker = "### BEGIN_PERFORMANCE_STATS ###"
    end_marker = "### END_PERFORMANCE_STATS ###"
    start = output.find(start_marker)
    end = output.find(end_marker, start)
    if start == -1 or end == -1:
        logger.error("Performance stats markers not found in output")
        raise RuntimeError("Performance stats markers not found")
    json_text = output[start + len(start_marker):end].strip()
    return json.loads(json_text)


def run_and_capture(cmd: list[str], log_file: Path) -> subprocess.CompletedProcess[str] | None:
    """
    Runs the command, captures stdout/stderr to the log_file.
    Returns the CompletedProcess on success, or None on failure.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Save stdout/stderr on failure and return None
        with open(log_file, "w") as lf:
            lf.write(e.stdout or "")
            lf.write(e.stderr or "")
        logger.error("Command failed: %s (see log: %s)", ' '.join(cmd), log_file)
        return None

    # Save stdout/stderr
    with open(log_file, "w") as lf:
        lf.write(proc.stdout or "")
        lf.write(proc.stderr or "")
    return proc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect tracer metrics")
    parser.add_argument(
        "--synthetic-bugs",
        action="store_true",
        help="Enable synthetic bug testing mode",
    )
    parser.add_argument(
        "--bug-wait-for-streams",
        action="store_true",
        help="Enable the wait_for_streams_thunk synthetic bug",
    )
    parser.add_argument(
        "--bug-collective-done",
        action="store_true",
        help="Enable the nccl_collective_done_thunk synthetic bug",
    )
    parser.add_argument(
        "--bug-remove-control-deps",
        action="store_true",
        help="Enable synthetic removal of collective control deps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting tracer performance collection")

    metrics_root = BUG_METRICS_ROOT if args.synthetic_bugs else METRICS_ROOT
    if args.bug_remove_control_deps:
        metrics_root = metrics_root / "remove_control_deps"
    elif args.bug_collective_done:
        metrics_root = metrics_root / "collective_done"
    elif args.bug_wait_for_streams:
        metrics_root = metrics_root / "wait_for_streams"

    run_indexes = range(1, 2) if args.synthetic_bugs else range(1, 4)

    for batch_dir in sorted(STABLEHLO_ROOT.glob("*_*")):
        if not batch_dir.is_dir():
            continue
        logger.info("Processing batch directory: %s", batch_dir)
        metrics_dir = metrics_root / batch_dir.name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        for hlo in batch_dir.glob("*.stablehlo"):
            task_name = hlo.stem
            logger.info("Processing HLO file: %s", hlo)
            for trace in (0, 1):
                for run_idx in run_indexes:
                    out_file = metrics_dir / f"{task_name}-trace{trace}.{run_idx}.json"
                    log_file = metrics_dir / f"{task_name}-trace{trace}.{run_idx}.log"

                    if out_file.exists():
                        logger.info("Skipping existing metrics: %s", out_file)
                        continue

                    cmd = [
                        "./bazel-bin/xla/tests/concurrency_trace/tracer_perf_main",
                        "--stablehlo=1",
                        f"--input={hlo}",
                        f"--trace={trace}",
                    ]
                    if args.synthetic_bugs:
                        if args.bug_wait_for_streams:
                            cmd.append("--bug_wait_for_streams=1")
                        if args.bug_collective_done:
                            cmd.append("--bug_collective_done=1")
                        if args.bug_remove_control_deps:
                            cmd.append("--bug_remove_control_deps=1")
                    logger.debug("Running command: %s", ' '.join(cmd))

                    proc = run_and_capture(cmd, log_file)
                    if proc is None:
                        # Command failed, skip JSON extraction
                        continue

                    # Extract metrics JSON
                    try:
                        stats = extract_json(proc.stdout or proc.stderr)
                    except RuntimeError as e:
                        logger.error(
                            "Failed to extract JSON for %s trace %d run %d: %s",
                            hlo, trace, run_idx, e,
                        )
                        continue

                    # Write JSON to file
                    with open(out_file, "w") as f:
                        json.dump(stats, f, indent=2)
                    logger.info("Wrote metrics to: %s", out_file)

    logger.info("Finished tracer performance collection")


if __name__ == "__main__":
    main()
