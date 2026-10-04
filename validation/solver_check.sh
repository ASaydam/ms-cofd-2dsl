#!/bin/bash
#
# Solves given meshes with the OpenFOAM of the container and the second-order case templates, to compare the
# results with those of another OpenFOAM version on the same meshes (see docs/validation.md). On the cluster:
#
#   bash validation/solver_check.sh <mesh folder> <output folder> <case list>
#
# <mesh folder> holds base/, 4x/ and 16x/ with <case>.msh files; one SLURM job per case and resolution, the 4x and
# 16x jobs start from the base solution (mapFields). Settings of the jobs: config/slurm.yaml, QoS of the current date.

set -euo pipefail
meshes=$(readlink -f "$1"); output=$(readlink -m "$2"); cases=$(readlink -f "$3")
source "$(dirname "$(readlink -f "$0")")/../hpc/env.sh"
qos=$("$MSCOFD_HOST_PYTHON" -c "import sys; sys.path.insert(0, '$MSCOFD_REPO/hpc'); import topup; print(topup.current_window(topup.load_settings()[0])['qos'])")
run="apptainer exec --bind $MSCOFD_BIND --home $MSCOFD_HOME/container/home $MSCOFD_IMAGE bash -c"

for resolution in base 4x 16x; do
    template="$MSCOFD_REPO/openfoam/case-templates/second-order/$resolution"
    while read -r case; do
        [ -z "$case" ] && continue
        folder="$output/$resolution/$case"
        if [ ! -d "$folder" ]; then
            mkdir -p "$output/$resolution"
            cp -r "$template" "$folder"
            cp "$meshes/$resolution/$case.msh" "$folder/mesh.msh"
            if [ "$resolution" != base ]; then echo "$output/base/$case" > "$folder/init_from"; fi
        fi
    done < "$cases"
done

while read -r case; do
    [ -z "$case" ] && continue
    common=(--parsable --account=users --partition=mid "--qos=$qos" --ntasks=1 --cpus-per-task=1 --mem=6G)
    base=$(sbatch "${common[@]}" --time=02:00:00 --job-name="m2d_check_base__$case" --chdir="$output/base/$case" \
        --output="$output/base/$case/%j.out" --wrap="$run 'source /opt/mscofd/activate.sh && bash $MSCOFD_REPO/openfoam/scripts/run_case.sh run'")
    for resolution in 4x 16x; do
        sbatch "${common[@]}" --time=20:00:00 --dependency="afterok:$base" --job-name="m2d_check_${resolution}__$case" \
            --chdir="$output/$resolution/$case" --output="$output/$resolution/$case/%j.out" \
            --wrap="$run 'source /opt/mscofd/activate.sh && bash $MSCOFD_REPO/openfoam/scripts/run_case.sh run'" > /dev/null
    done
    echo "$case: base job $base, 4x and 16x submitted"
done < "$cases"
