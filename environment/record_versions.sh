#!/bin/bash
# Writes the versions of all software in the container to a folder (default /opt/mscofd/versions):
#   summary.txt         OpenFOAM, gmsh, VTK, Python, micromamba, OS
#   conda-explicit.txt  every conda package with its exact build (conda list --explicit)
#   pip-freeze.txt      Python packages as pip sees them
set -e
output=${1:-/opt/mscofd/versions}
mkdir -p "$output"
source /opt/mscofd/activate.sh

{
    echo "OpenFOAM: $(foamVersion 2> /dev/null || echo "${WM_PROJECT_VERSION:-unknown}") (WM_PROJECT_VERSION=${WM_PROJECT_VERSION:-}, API ${FOAM_API:-})"
    echo "gmsh: $(gmsh --version 2>&1 | tail -1)"
    echo "vtk: $(python -c 'import vtk; print(vtk.vtkVersion.GetVTKVersion())')"
    echo "python: $(python --version 2>&1)"
    echo "numpy: $(python -c 'import numpy; print(numpy.__version__)')"
    echo "h5py: $(python -c 'import h5py; print(h5py.__version__)')"
    echo "rclone: $(rclone version 2>&1 | head -1)"
    echo "micromamba: $(micromamba --version)"
    echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
} > "$output/summary.txt"

micromamba list -p /opt/conda/envs/mscofd --explicit --md5 > "$output/conda-explicit.txt"
python -m pip freeze > "$output/pip-freeze.txt" 2> /dev/null || true
cat "$output/summary.txt"
