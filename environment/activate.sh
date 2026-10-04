# Activates the software of the MS-COFD-2DSL container: OpenFOAM and the Python environment.
# Sourced by the pipeline scripts that run inside the container (apptainer exec ... bash -c "source ...; ...").

# OpenFOAM's bashrc reads unset variables and returns non-zero codes, it is sourced without "set -eu"; the shell
# options of the caller are restored afterwards
_mscofd_options=$(set +o)
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc > /dev/null 2>&1
eval "$_mscofd_options"
unset _mscofd_options

export PATH="/opt/conda/envs/mscofd/bin:$PATH"
export MSCOFD_VERSIONS=/opt/mscofd/versions
