# Environment of the MS-COFD-2DSL production on KUACC, sourced by the host-side scripts and every job.
#
# Cluster rule: nothing is created next to hpc_run, so the data, the container, the caches and the configuration
# all live under hpc_run/ms-cofd-2dsl (MSCOFD_DATA_ROOT can be set before sourcing, e.g. for the pilot):
#
#   repo/        code (git archive of a committed revision, REVISION holds the commit)
#   container/   images (current: path of the image in use), Apptainer cache, home folder of the container
#   config/      rclone.conf, remote (rclone target of the archives)
#   geometry/  state/  work/  seeds/  results/  archives/  logs/  dataset/

# real paths (hpc_run in the home folder is a link to /scratch; only /scratch is bound into the container)
MSCOFD_HPC_RUN=$(readlink -f "${MSCOFD_HPC_RUN:-/kuacc/users/$USER/hpc_run}")
MSCOFD_HOME=$(readlink -f "${MSCOFD_HOME:-$MSCOFD_HPC_RUN/ms-cofd-2dsl}")
MSCOFD_DATA_ROOT=$(readlink -m "${MSCOFD_DATA_ROOT:-$MSCOFD_HOME}")
MSCOFD_REPO=$(readlink -f "${MSCOFD_REPO:-$MSCOFD_HOME/repo}")
export MSCOFD_HPC_RUN MSCOFD_HOME MSCOFD_DATA_ROOT MSCOFD_REPO
export MSCOFD_BIND=$MSCOFD_HPC_RUN

# empty when not set up yet (the scripts that source this file may use "set -e"); a remote set before sourcing
# (e.g. a separate folder for a pilot) is kept
MSCOFD_IMAGE=$(cat "$MSCOFD_HOME/container/current" 2> /dev/null || true)
MSCOFD_REMOTE=${MSCOFD_REMOTE:-$(cat "$MSCOFD_HOME/config/remote" 2> /dev/null || true)}
export MSCOFD_IMAGE MSCOFD_REMOTE
export MSCOFD_HOST_PYTHON=${MSCOFD_HOST_PYTHON:-/kuacc/apps/anaconda/2024.02/bin/python3}

export APPTAINER_CACHEDIR=$MSCOFD_HOME/container/cache
export APPTAINER_TMPDIR=$MSCOFD_HOME/container/tmp
export APPTAINER_CONFIGDIR=$MSCOFD_HOME/container/config   # otherwise Apptainer creates ~/.apptainer
export RCLONE_CONFIG=$MSCOFD_HOME/config/rclone.conf
export XDG_CACHE_HOME=$MSCOFD_HOME/cache
export MPLCONFIGDIR=$MSCOFD_HOME/cache/matplotlib

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CONFIGDIR" "$MSCOFD_HOME/container/home" \
    "$MSCOFD_HOME/config" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$MSCOFD_DATA_ROOT/state" "$MSCOFD_DATA_ROOT/logs"

module load apptainer/1.4.1 2> /dev/null || true
