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
invokes ``tracer_perf_main`` for each ``.stablehlo`` file with tracing
both disabled and enabled, and stores the extracted JSON metrics and
command logs in a mirrored directory tree under ``modelgarden/nlp/metrics``.
"""

from __future__ import annotations

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


def main() -> None:
    logger.info("Starting tracer performance collection")

    for batch_dir in sorted(STABLEHLO_ROOT.glob("*_*")):
        if not batch_dir.is_dir():
            continue
        logger.info("Processing batch directory: %s", batch_dir)
        metrics_dir = METRICS_ROOT / batch_dir.name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        for hlo in batch_dir.glob("*.stablehlo"):
            task_name = hlo.stem
            logger.info("Processing HLO file: %s", hlo)
            for trace in (0, 1):
                for run_idx in range(1, 4):
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
