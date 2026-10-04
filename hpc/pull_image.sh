#!/bin/bash
#
# Pulls the pipeline container with Apptainer and makes it the image of the production (on the cluster):
#
#   bash hpc/pull_image.sh sha256:<digest>
#
# The image is pulled by digest (ghcr.io/asaydam/ms-cofd-2dsl@sha256:...), so the exact image is known; the path of
# the .sif file is written to container/current, which hpc/env.sh reads. Images are kept, an older one can be made
# current again by writing its path to container/current.

set -euo pipefail
digest=$1
source "$(dirname "$(readlink -f "$0")")/env.sh"

short=${digest#sha256:}
image="$MSCOFD_HOME/container/ms-cofd-2dsl_${short:0:12}.sif"
if [ ! -f "$image" ]; then
    apptainer pull "$image" "docker://ghcr.io/asaydam/ms-cofd-2dsl@$digest"
fi
echo "$image" > "$MSCOFD_HOME/container/current"
echo "$digest" > "$MSCOFD_HOME/container/current_digest"
apptainer exec "$image" cat /opt/mscofd/versions/summary.txt
echo "current image: $image"
