#!/usr/bin/env bash

set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="${PY_SCRIPT:-main.py}"
CONFIG="${1:-configs/blinkflow_human.yaml}"
RESUME_STATE="${RESUME_STATE:-resume_state.yaml}"
RESTART_DELAY="${RESTART_DELAY:-5}"

shift || true

echo "Using config: ${CONFIG}"
echo "Using Python script: ${PY_SCRIPT}"
echo "Using resume file: ${RESUME_STATE}"
echo

is_dataset_complete() {
"${PYTHON_BIN}" - <<PY
import sys
import yaml
import importlib.util
from pathlib import Path

py_script = Path("${PY_SCRIPT}")
config_path = Path("${CONFIG}")
resume_path = Path("${RESUME_STATE}")

if not py_script.exists():
    print(f"Python script not found: {py_script}")
    sys.exit(2)

if not config_path.exists():
    print(f"Config file not found: {config_path}")
    sys.exit(2)

with open(config_path, "r") as f:
    config = yaml.safe_load(f) or {}

# Import build_jobs() from your main.py without running main()
spec = importlib.util.spec_from_file_location("dataset_main", py_script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

jobs = mod.build_jobs(config)
expected = {job["folder_name"] for job in jobs}

if resume_path.exists():
    with open(resume_path, "r") as f:
        resume = yaml.safe_load(f) or {}
else:
    resume = {}

state = resume.get("resume_state", {})
completed = set(state.get("completed_jobs", []))

missing = sorted(expected - completed)

print(f"Completed jobs: {len(completed & expected)} / {len(expected)}")

if missing:
    print("Still missing jobs:")
    for name in missing[:20]:
        print(f"  - {name}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    sys.exit(1)

print("All selected jobs are complete.")
sys.exit(0)
PY
}

attempt=1

while true; do
    echo "Checking completion state..."
    if is_dataset_complete; then
        echo
        echo "Dataset generation is complete. Stopping."
        exit 0
    fi

    echo
    echo "Starting dataset generation attempt #${attempt}..."
    echo "Command: ${PYTHON_BIN} ${PY_SCRIPT} --config ${CONFIG} $*"
    echo

    "${PYTHON_BIN}" "${PY_SCRIPT}" --config "${CONFIG}" "$@"
    exit_code=$?

    echo
    echo "Python script exited with code: ${exit_code}"

    echo
    echo "Re-checking completion state..."
    if is_dataset_complete; then
        echo
        echo "Dataset generation is complete. Stopping."
        exit 0
    fi

    echo
    echo "Dataset is not complete yet. Restarting after ${RESTART_DELAY} seconds..."
    sleep "${RESTART_DELAY}"

    attempt=$((attempt + 1))
done