# Activates the software of the MS-COFD-2DSL container: OpenFOAM and the Python environment.
# Sourced by the pipeline scripts that run inside the container (apptainer exec ... bash -c "source ...; ...").

# OpenFOAM's bashrc reads unset variables, so it is sourced without "set -u"
_mscofd_nounset=$([[ $- == *u* ]] && echo 1 || true)
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
[ -n "$_mscofd_nounset" ] && set -u
unset _mscofd_nounset

export PATH="/opt/conda/envs/mscofd/bin:$PATH"
export MSCOFD_VERSIONS=/opt/mscofd/versions
