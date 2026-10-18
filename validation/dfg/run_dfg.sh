#!/bin/bash
#
# One mesh level of the DFG 2D-1 benchmark (validation/dfg/dfg.py, docs/validation.md), in the container after
# /opt/mscofd/activate.sh:
#
#   bash validation/dfg/run_dfg.sh <root> <level> [<source level>]
#
# <root>/<level>: setup (production template, mesh.geo), mesh (gmsh -> gmshToFoam -> changeDictionary -> checkMesh,
# with the gmsh option variants of config/dataset.yaml when checkMesh fails), the initial fields mapped from the
# solution of <source level> when given (mapFields, as the production starts its fine runs), the parabolic inlet,
# simpleFoam until residualControl stops it, then the function objects of the benchmark on the last time step.

set -euo pipefail
root=$1
level=$2
source_level=${3:-}
repo=$(cd "$(dirname "$0")/../.." && pwd)
case=$root/$level

python "$repo/validation/dfg/dfg.py" setup "$case" "$(python "$repo/validation/dfg/dfg.py" size "$level")"
cd "$case"
cp mesh.geo mesh.geo.base
number=0
for options in '' 'Mesh.Smoothing = 5;' 'Mesh.Smoothing = 10;' 'Mesh.RecombinationAlgorithm = 0;'; do
    { cat mesh.geo.base; [ -n "$options" ] && printf '\n// mesh variant %s\n%s\n' "$number" "$options"; } > mesh.geo || true
    rm -rf constant/polyMesh
    gmsh -3 mesh.geo -o mesh.msh -format msh2 -nt 1 > log.gmsh 2>&1
    gmshToFoam mesh.msh > log.gmshToFoam 2>&1
    changeDictionary > log.changeDictionary 2>&1
    checkMesh > log.checkMesh 2>&1 || true
    if grep -q "Mesh OK." log.checkMesh; then
        echo "$number $options" > mesh_variant
        break
    fi
    cp log.checkMesh "log.checkMesh.variant$number"
    number=$((number + 1))
done
grep -q "Mesh OK." log.checkMesh

if [ -n "$source_level" ]; then
    mapFields "$root/$source_level" -consistent -sourceTime latestTime -mapMethod cellPointInterpolate \
        > log.mapFields 2>&1
    echo "$root/$source_level" > mapped
fi
postProcess -func writeCellCentres -time 0 > log.writeCellCentres 2>&1
python "$repo/validation/dfg/dfg.py" inlet "$case"

simpleFoam > log.simpleFoam 2>&1
python "$repo/validation/dfg/dfg.py" functions "$case"
simpleFoam -postProcess -latestTime > log.postProcess 2>&1
echo "$level done: $(grep -c '^Time = ' log.simpleFoam) iterations"
