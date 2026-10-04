"""
Exports the flow fields of the OpenFOAM cases to CSV files on a regular grid, with ParaView.

Run it with a Python that has ParaView: the python of an environment with the paraview package (the released
extracts were made this way, from Jupyter), or pvpython of a ParaView installation:

    python extraction/extract_fields.py outputs/cases/base outputs/extracts/base --dims 260 120
    python extraction/extract_fields.py outputs/cases/16x outputs/extracts/16x --dims 1040 480

Use a shell where OpenFOAM is not loaded: OpenFOAM's environment adds its own ParaView to PATH and
LD_LIBRARY_PATH, and those libraries clash with the ParaView used here.

For every case the last written time step (the converged solution) is written to <case>_last_time.csv, with the
columns U:0, U:1, U:2, p and vtkValidPointMask (0 for grid points inside the obstacle, which are outside of the
mesh). The geometry channels of the dataset are built from this mask (see build_arrays.py), so no other time step
is needed. The exported time of every case is written to extraction_log.csv, cases that already have a
_last_time.csv file are skipped.
"""

import argparse
import csv
import glob
import os
import sys

# the readers and filters of paraview.simple are created when ParaView starts its session, starting it fails
# when ParaView finds the libraries of another VTK / ParaView
try:
    from paraview.simple import OpenFOAMReader, ResampleToImage, SaveData, GetAnimationScene, UpdatePipeline, Delete
except Exception as error:
    hint = ""
    if os.environ.get("WM_PROJECT_DIR"):
        hint += (f"\nOpenFOAM is loaded in this shell ({os.environ['WM_PROJECT_DIR']}), it adds its own ParaView to "
                 f"PATH and LD_LIBRARY_PATH. Run this script in a new shell without OpenFOAM.")
    try:
        from importlib.metadata import version
        hint += (f"\nThe pip package vtk {version('vtk')} is installed in this environment (e.g. for pyvista), its "
                 f"vtkmodules replace the VTK of ParaView. Use an environment with ParaView but without pip vtk.")
    except Exception:
        pass
    sys.exit(f"ParaView could not be started: {type(error).__name__}: {error}{hint}")

LOG_FIELDS = ['case_name', 'status', 'last_time', 'number_of_times']


def extract_data(case_path, case_name, save_path, dims, foam_file="foam.foam"):
    """
    resamples the last written time step of a case to a regular grid and saves it as CSV
    :param case_path: OpenFOAM case folder
    :param case_name: name used for the output file
    :param save_path: output folder
    :param dims: number of grid points in x and y
    :return: exported time and the number of written time steps
    """
    # the reader needs an (empty) foam.foam file in the case folder
    file_name = os.path.join(case_path, foam_file)
    if not os.path.exists(file_name):
        open(file_name, 'w').close()

    # create a new 'OpenFOAMReader'
    reader = OpenFOAMReader(registrationName=foam_file, FileName=file_name)
    reader.MeshRegions = ['internalMesh']
    reader.CellArrays = ['U', 'p']
    reader.SkipZeroTime = 1  # the 0 folder is the initial condition
    reader.UpdatePipelineInformation()

    times = list(reader.TimestepValues)
    if not times:
        Delete(reader)
        raise RuntimeError("no written time steps found")
    last_time = times[-1]

    # create a new 'Resample To Image', the grid covers the whole mesh (the domain, in meters)
    resample = ResampleToImage(registrationName='ResampleToImage1', Input=reader)
    resample.UseInputBounds = 1
    resample.SamplingDimensions = [dims[0], dims[1], 1]

    # set the animation time as well, SaveData writes the data of the current animation time
    animation_scene = GetAnimationScene()
    animation_scene.UpdateAnimationUsingDataTimeSteps()

    animation_scene.AnimationTime = last_time
    UpdatePipeline(time=last_time, proxy=resample)

    # save data
    SaveData(os.path.join(save_path, f"{case_name}_last_time.csv"), proxy=resample, ChooseArraysToWrite=1,
             PointDataArrays=['U', 'p', 'vtkValidPointMask'])

    Delete(resample)
    Delete(reader)

    return last_time, len(times)


def read_log(log_file):
    if not os.path.exists(log_file):
        return {}
    with open(log_file, 'r', newline='') as file:
        return {row['case_name']: row for row in csv.DictReader(file)}


def write_log(rows, log_file):
    with open(log_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: row['case_name']))


def main():
    parser = argparse.ArgumentParser(description="Export OpenFOAM results to CSV files with ParaView")
    parser.add_argument("cases_dir", help="folder of the OpenFOAM cases of one resolution, e.g. outputs/cases/base")
    parser.add_argument("save_path", help="output folder, e.g. outputs/extracts/base")
    parser.add_argument("--dims", type=int, nargs=2, default=[260, 120],
                        help="grid points in x and y: 260 120 for base, 1040 480 for 16x")
    parser.add_argument("--overwrite", action="store_true", help="export the cases that were exported before")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    log_file = os.path.join(args.save_path, "extraction_log.csv")
    log_rows = read_log(log_file)

    cases = sorted(path for path in glob.glob(os.path.join(args.cases_dir, "*")) if os.path.isdir(path))
    print(f"There are {len(cases)} cases, extract path: {args.save_path}")

    for case in cases:
        case_name = os.path.basename(case)
        if os.path.exists(os.path.join(args.save_path, case_name + "_last_time.csv")) and not args.overwrite:
            continue

        try:
            last_time, number_of_times = extract_data(case, case_name, args.save_path, args.dims)
            log_rows[case_name] = {'case_name': case_name, 'status': 'ok', 'last_time': last_time,
                                   'number_of_times': number_of_times}
            print(f"{case_name}: exported time {last_time}")
        except Exception as error:
            log_rows[case_name] = {'case_name': case_name, 'status': f'failed: {error}'}
            print(f"{case_name}: failed ({error})")

        # write the log after every case, so it is up to date if the run is interrupted
        write_log(log_rows, log_file)

    write_log(log_rows, log_file)


if __name__ == "__main__":
    main()
