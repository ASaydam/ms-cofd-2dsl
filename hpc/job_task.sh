#!/bin/bash
#
# One task of the production in a SLURM job, submitted by hpc/topup.py:
#
#   sbatch ... hpc/job_task.sh <run> <case>
#
# Runs hpc/run_task.py in the container (Apptainer), then tops up the queue. SLURM settings come from topup.py
# (config/slurm.yaml); the environment (data root, image, remote) from hpc/env.sh.

set -uo pipefail
run=$1
case=$2

source "$MSCOFD_REPO/hpc/env.sh"

"$MSCOFD_HOST_PYTHON" "$MSCOFD_REPO/hpc/ledger.py" set "$run" "$case" running "${SLURM_JOB_ID:-} $(hostname -s)"

# CPUs of the node allocated when the task starts (for the cost study: how many jobs shared the node)
MSCOFD_NODE_ALLOCATED=$(scontrol show node "$(hostname -s)" 2> /dev/null | grep -o "CPUAlloc=[0-9]*" | cut -d= -f2)
export MSCOFD_NODE_ALLOCATED

apptainer exec --bind "$MSCOFD_BIND" --home "$MSCOFD_DATA_ROOT/container/home" "$MSCOFD_IMAGE" \
    bash -c 'source /opt/mscofd/activate.sh && cd "$MSCOFD_REPO" && python -m hpc.run_task "$0" "$1"' "$run" "$case"
status=$?

# the task sets its state; a container that did not start leaves it running, topup.py submits it again
"$MSCOFD_HOST_PYTHON" "$MSCOFD_REPO/hpc/topup.py" || true
exit $status
