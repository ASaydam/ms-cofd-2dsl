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

# one job per task: the job holds <data root>/state/<run>/<case>.lock (a directory: mkdir is atomic on the shared
# file system, flock is not across nodes). A second job of the same task stops without touching it; a lock left by
# a job that is gone (killed, or this job before a requeue) is taken over.
lock="$MSCOFD_DATA_ROOT/state/$run/$case.lock"
mkdir -p "$(dirname "$lock")"
if ! mkdir "$lock" 2> /dev/null; then
    holder=""
    for _ in 1 2 3 4 5 6 7 8 9 10; do  # the holder writes its job id right after mkdir
        holder=$(cat "$lock/job" 2> /dev/null)
        [ -n "$holder" ] && break
        sleep 1
    done
    if [ -n "$holder" ] && [ "$holder" != "${SLURM_JOB_ID:-}" ] && squeue -h -j "$holder" 2> /dev/null | grep -q .; then
        echo "$run $case is run by job $holder, job ${SLURM_JOB_ID:-} stops"
        exit 0
    fi
    rm -rf "$lock"
    mkdir "$lock" || exit 1
fi
echo "${SLURM_JOB_ID:-$$}" > "$lock/job"

"$MSCOFD_HOST_PYTHON" "$MSCOFD_REPO/hpc/ledger.py" set "$run" "$case" running "${SLURM_JOB_ID:-} $(hostname -s)"

# CPUs of the node allocated when the task starts (for the cost study: how many jobs shared the node)
MSCOFD_NODE_ALLOCATED=$(scontrol show node "$(hostname -s)" 2> /dev/null | grep -o "CPUAlloc=[0-9]*" | cut -d= -f2)
export MSCOFD_NODE_ALLOCATED

apptainer exec --bind "$MSCOFD_BIND" --home "$MSCOFD_HOME/container/home" "$MSCOFD_IMAGE" \
    bash -c 'source /opt/mscofd/activate.sh && cd "$MSCOFD_REPO" && python -m hpc.run_task "$0" "$1"' "$run" "$case"
status=$?
rm -rf "$lock"

# the task sets its state; a container that did not start leaves it running, topup.py submits it again
"$MSCOFD_HOST_PYTHON" "$MSCOFD_REPO/hpc/topup.py" || true
exit $status
