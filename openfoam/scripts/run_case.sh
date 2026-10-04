#!/bin/bash
#
# Runs one OpenFOAM case: gmsh mesh -> OpenFOAM mesh -> checkMesh -> (mapFields) -> simpleFoam.
#
# It is submitted by submit_all.sh / submit_pairs.sh / resubmit.sh (sbatch --chdir=<case folder>), or run by hand
# inside a case folder:
#
#   bash run_case.sh                  # convert the mesh and run the solver
#   bash run_case.sh rerun            # clean the case first, then the same as above
#   bash run_case.sh continue 5000    # continue from the latest time step until iteration 5000
#
#SBATCH --job-name=mscofd_case
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%j-openfoam.out

mode=${1:-run}
end_time=${2:-}

# load OpenFOAM when a module is given (see slurm_settings.sh), otherwise it has to be loaded already
if [ -n "${OPENFOAM_MODULE:-}" ]; then
    module load "$OPENFOAM_MODULE"
    if [ -n "${FOAM_BASH:-}" ]; then source "$FOAM_BASH"; fi
fi

# stop at the first failing command, the job then fails with the exit code of that command
set -e

# set_control_dict <keyword> <value> : changes an entry of system/controlDict
set_control_dict() {
    if command -v foamDictionary > /dev/null; then
        foamDictionary system/controlDict -entry "$1" -set "$2" > /dev/null
    else
        sed -i "s/^\($1[[:space:]]\+\)[^;]*;/\1$2;/" system/controlDict
    fi
}

# record_run_info : adds a line to run_info.csv about where this run happens (host, CPU model, CPUs of the node and
# how many of them are allocated when the run starts), for comparing run times (read by extraction/check_runs.py)
record_run_info() {
    local cpu_model node_cpus node_allocated
    cpu_model=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: *//; s/,/ /g' || true)
    node_cpus=$(nproc --all)
    node_allocated=$(scontrol show node "$(hostname -s)" 2> /dev/null | grep -o "CPUAlloc=[0-9]*" | cut -d= -f2 || true)
    if [ ! -f run_info.csv ]; then
        echo "mode,start,host,cpu_model,node_cpus,node_cpus_allocated,slurm_job_id" > run_info.csv
    fi
    echo "$mode,$(date -Iseconds),$(hostname -s),$cpu_model,$node_cpus,$node_allocated,${SLURM_JOB_ID:-}" >> run_info.csv
}

echo "Running case $(basename "$PWD") ($mode)"
record_run_info

case "$mode" in
    run|rerun)
        # clean case (time folders, polyMesh and logs of the previous run, mesh.msh is kept)
        if [ "$mode" = "rerun" ]; then
            foamCleanTutorials
        fi

        # convert gmsh mesh to openfoam mesh and set the patch types (system/changeDictionaryDict)
        gmshToFoam mesh.msh > log.gmshToFoam 2>&1
        changeDictionary > log.changeDictionary 2>&1

        # check mesh quality, problems are reported but the case is still run
        checkMesh > log.checkMesh 2>&1 || true
        if ! grep -q "Mesh OK." log.checkMesh; then
            echo "checkMesh reported problems, see log.checkMesh"
        fi

        # start from the latest solution of another case with the same geometry (written by create_foam.py),
        # otherwise from the uniform fields of the 0 folder
        if [ -f init_from ]; then
            mapFields "$(cat init_from)" -consistent -sourceTime latestTime -mapMethod cellVolumeWeight \
                > log.mapFields 2>&1
        fi

        # run solver
        simpleFoam > log.simpleFoam 2>&1
        ;;

    continue)
        set_control_dict startFrom latestTime
        if [ -n "$end_time" ]; then
            set_control_dict endTime "$end_time"
        fi

        # run solver, the log of the first run is kept
        simpleFoam >> log.simpleFoam 2>&1
        ;;

    *)
        echo "Unknown mode '$mode', use run, rerun or continue"
        exit 1
        ;;
esac

# execution time of the solver
grep "ExecutionTime = " log.simpleFoam | tail -1
echo "Case finished"
