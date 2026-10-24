#!/bin/bash
#
# One case of the tolerance check (validation/tolerance_check.py, docs/validation.md) in a SLURM job:
#
#   export MSCOFD_DATA_ROOT=<validation root> MSCOFD_PRODUCTION_ROOT=<production root> MSCOFD_TOL_SBATCH="<options>"
#   sbatch --job-name=m2k_<case> --output=<validation root>/logs/%x-%j.out --export=ALL $MSCOFD_TOL_SBATCH \
#       validation/tolerance_check.sh <case> [<extra iterations>]
#
# MSCOFD_TOL_SBATCH: the SLURM options (default below). simpleFoam is stopped MSCOFD_STOP_MARGIN_S seconds (default
# 1800) before the time limit; the job then submits the same command again with these options, and the next job
# continues from the last written time step.
set -uo pipefail
case=$1
extra=${2:-30000}
options=${MSCOFD_TOL_SBATCH:---account=users --partition=long --qos=users --ntasks=1 --cpus-per-task=1 --mem=4G --time=4-00:00:00 --exclude=dy02,dy03}
source "$MSCOFD_REPO/hpc/env.sh"
source "$MSCOFD_REPO/hpc/time_left.sh"
# the image from a copy on the local disk of the node, as hpc/job_task.sh
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
left=$(seconds_left "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2> /dev/null)")
margin=${MSCOFD_STOP_MARGIN_S:-1800}
stop=()
if [ -n "$left" ] && [ "$left" -gt $((margin + 600)) ]; then
    stop=(--stop-after $((left - margin)))
fi
echo "$case: job ${SLURM_JOB_ID:-} on $(hostname -s), revision $(cat "$MSCOFD_REPO/REVISION"), ${left:-?} s left, $(date -Iseconds)"
apptainer exec --bind "$MSCOFD_BIND" --home "$MSCOFD_HOME/container/home" "$MSCOFD_IMAGE" \
    bash -c 'source /opt/mscofd/activate.sh && cd "$MSCOFD_REPO" && python validation/tolerance_check.py run "$@"' \
    _ "$case" --extra "$extra" "${stop[@]}"
status=$?
echo "$case: exit $status $(date -Iseconds)"
if [ $status -eq 3 ]; then
    # shellcheck disable=SC2086  # the options are separate words
    sbatch --job-name="${SLURM_JOB_NAME:-m2k_$case}" --output="$MSCOFD_DATA_ROOT/logs/%x-%j.out" --export=ALL \
        $options "$MSCOFD_REPO/validation/tolerance_check.sh" "$case" "$extra"
fi
exit $status
