# Sourced by hpc/job_task.sh (and tests/test_job_task.py):
#
#   seconds_left "$(squeue -h -j "$SLURM_JOB_ID" -o %L)"
#
# prints the time left of a job in seconds, from squeue's [days-]hours:minutes:seconds, minutes:seconds or seconds;
# prints nothing for UNLIMITED, NOT_SET, INVALID or an empty value (the job then runs without a stop of its own).

seconds_left() {
    local value=$1 days=0 seconds=0 part
    local -a parts
    case "$value" in
        '' | *[!0-9:-]*) return 0 ;;
    esac
    if [[ $value == *-* ]]; then
        days=${value%%-*}
        value=${value#*-}
    fi
    IFS=: read -r -a parts <<< "$value"
    for part in "${parts[@]}"; do
        seconds=$((seconds * 60 + 10#$part))
    done
    echo $((10#$days * 86400 + seconds))
}
