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

# the image is read from a copy on the local disk of the node (/tmp), made once per node and checked against
# container/current_md5: many jobs reading the image file on BeeGFS saw "cannot read file data: Input/output error"
image_md5=$(cat "$MSCOFD_HOME/container/current_md5" 2> /dev/null)
local_image="/tmp/mscofd-$USER/$(basename "$MSCOFD_IMAGE")"
if [ -n "$image_md5" ]; then
    mkdir -p "/tmp/mscofd-$USER"
    (
        flock -x 9
        if [ ! -f "$local_image.ok" ] || [ ! -f "$local_image" ]; then
            rm -f "$local_image" "$local_image.ok"
            cp "$MSCOFD_IMAGE" "$local_image.part" &&
                [ "$(md5sum < "$local_image.part" | cut -d' ' -f1)" = "$image_md5" ] &&
                mv "$local_image.part" "$local_image" && touch "$local_image.ok"
            rm -f "$local_image.part"
        fi
    ) 9> "/tmp/mscofd-$USER/copy.lock"
    [ -f "$local_image.ok" ] && [ -f "$local_image" ] && MSCOFD_IMAGE=$local_image
fi
export MSCOFD_IMAGE

# one line per job in state/<run>/<case>.jobs (cost study: queue wait, whole job, failed attempts; sacct is not
# available to users): "start <job> <node> SubmitTime=... StartTime=..." now, "end <job> <exit status> <time>" at the end
jobs_log="$MSCOFD_DATA_ROOT/state/$run/$case.jobs"
slurm_times=$(scontrol show job "${SLURM_JOB_ID:-0}" 2> /dev/null | grep -oE "(SubmitTime|StartTime)=[^ ]+" | tr '\n' ' ')
echo "start ${SLURM_JOB_ID:-} $(hostname -s) ${slurm_times}$(date -Iseconds)" >> "$jobs_log"

# a node where the container cannot start (e.g. "can't open /proc/self/mountinfo") would take every task submitted to
# it: after three tries the node is written to state/bad_nodes (topup.py excludes it for bad_node_hours), this
# submission is not counted, and the task is submitted again elsewhere. More than 5 such nodes point to a general
# problem: then the task goes the normal way (interrupted, submitted again up to max_attempts).
bad_nodes="$MSCOFD_DATA_ROOT/state/bad_nodes"
started=0
for try in 1 2 3; do
    if check=$(apptainer exec --bind "$MSCOFD_BIND" "$MSCOFD_IMAGE" true 2>&1); then started=1; break; fi
    [ $try -lt 3 ] && sleep 30
done
if [ $started -eq 0 ]; then
    if [ "$(wc -l < "$bad_nodes" 2> /dev/null || echo 0)" -lt 5 ]; then
        echo "$(hostname -s) $(date -Iseconds) $(echo "$check" | grep -m1 -E "FATAL|error" | cut -c1-200)" >> "$bad_nodes"
        attempts="$MSCOFD_DATA_ROOT/state/$run/$case.attempts"
        [ -f "$attempts" ] && echo $(( $(cat "$attempts") - 1 )) > "$attempts"
    fi
    echo "end ${SLURM_JOB_ID:-} container $(date -Iseconds)" >> "$jobs_log"
    echo "the container does not start on $(hostname -s): $check"
    rm -rf "$lock"
    "$MSCOFD_HOST_PYTHON" "$MSCOFD_REPO/hpc/topup.py" || true
    exit 1
fi

"$MSCOFD_HOST_PYTHON" "$MSCOFD_REPO/hpc/ledger.py" set "$run" "$case" running "${SLURM_JOB_ID:-} $(hostname -s)"

# CPUs of the node allocated when the task starts (for the cost study: how many jobs shared the node)
MSCOFD_NODE_ALLOCATED=$(scontrol show node "$(hostname -s)" 2> /dev/null | grep -o "CPUAlloc=[0-9]*" | cut -d= -f2)
export MSCOFD_NODE_ALLOCATED

apptainer exec --bind "$MSCOFD_BIND" --home "$MSCOFD_HOME/container/home" "$MSCOFD_IMAGE" \
    bash -c 'source /opt/mscofd/activate.sh && cd "$MSCOFD_REPO" && python -m hpc.run_task "$0" "$1"' "$run" "$case"
status=$?
echo "end ${SLURM_JOB_ID:-} $status $(date -Iseconds)" >> "$jobs_log"
rm -rf "$lock"

# the task sets its state; a container that did not start leaves it running, topup.py submits it again
"$MSCOFD_HOST_PYTHON" "$MSCOFD_REPO/hpc/topup.py" || true
exit $status
