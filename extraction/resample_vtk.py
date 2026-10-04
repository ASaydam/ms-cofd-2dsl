"""
Resamples the flow fields of an OpenFOAM case to a regular grid with VTK.

This replaces the ParaView export of MS-COFD (v4) with the same VTK filters ParaView used, without ParaView:

    vtkOpenFOAMReader   internal mesh, cell arrays U and p, cell-to-point interpolation, 0 folder skipped
    vtkResampleToImage  grid over the bounds of the mesh (UseInputBounds), nx x ny x 1 points

The grid points are spread over the whole domain, both ends included (360 x 120 points over 0-360 mm and 0-120 mm
for the base resolution). Grid points inside the obstacle are outside of the mesh: their vtkValidPointMask is 0 and
their values are 0, as in the ParaView export.

    python -m extraction.resample_vtk <case folder> --dims 360 120            # prints a summary
"""

import argparse
from pathlib import Path

import numpy as np


def latest_time(case_folder):
    """
    largest written time step of a case (the 0 folder is the initial condition and not counted)
    """
    times = []
    for path in Path(case_folder).iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and value > 0:
            times.append(value)
    if not times:
        raise RuntimeError(f"{case_folder}: no written time steps found")
    return max(times)


def resample_case(case_folder, nx, ny, foam_file="foam.foam"):
    """
    resamples the last written time step of a case to a regular grid
    :param case_folder: OpenFOAM case folder
    :param nx: number of grid points along the flow
    :param ny: number of grid points across the channel
    :return: dictionary with u_x, u_y, p and the valid point mask as (nx, ny) arrays, and the exported time
    """
    # imported here, the rest of the pipeline does not need VTK
    from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader
    from vtkmodules.vtkFiltersCore import vtkResampleToImage
    from vtkmodules.util.numpy_support import vtk_to_numpy

    case_folder = Path(case_folder)
    foam_path = case_folder / foam_file
    if not foam_path.exists():
        foam_path.touch()  # the reader needs an (empty) .foam file in the case folder

    reader = vtkOpenFOAMReader()
    reader.SetFileName(str(foam_path))
    reader.SkipZeroTimeOn()            # the 0 folder is the initial condition
    reader.CreateCellToPointOn()       # point data from the cell data, as ParaView's reader does by default
    reader.UpdateInformation()

    # internal mesh only, cell arrays U and p only
    for i in range(reader.GetNumberOfPatchArrays()):
        name = reader.GetPatchArrayName(i)
        reader.SetPatchArrayStatus(name, 1 if name == 'internalMesh' else 0)
    reader.DisableAllCellArrays()
    reader.SetCellArrayStatus('U', 1)
    reader.SetCellArrayStatus('p', 1)

    time = latest_time(case_folder)
    reader.UpdateTimeStep(time)
    reader.Update()

    resample = vtkResampleToImage()
    resample.SetInputConnection(reader.GetOutputPort())
    resample.UseInputBoundsOn()
    resample.SetSamplingDimensions(nx, ny, 1)
    resample.UpdateTimeStep(time)
    resample.Update()

    point_data = resample.GetOutput().GetPointData()

    def field(name, component=None):
        values = vtk_to_numpy(point_data.GetArray(name))
        if component is not None:
            values = values[:, component]
        # the points go along x first, then y
        return np.asarray(values, dtype=float).reshape(ny, nx).T

    return {'u_x': field('U', 0), 'u_y': field('U', 1), 'p': field('p'), 'valid': field('vtkValidPointMask'),
            'time': time}


def main():
    parser = argparse.ArgumentParser(description="Resample the last time step of an OpenFOAM case with VTK")
    parser.add_argument("case_folder")
    parser.add_argument("--dims", type=int, nargs=2, default=[360, 120], help="grid points in x and y")
    args = parser.parse_args()

    fields = resample_case(args.case_folder, *args.dims)
    obstacle = fields['valid'] == 0
    print(f"time {fields['time']}, grid {args.dims[0]} x {args.dims[1]}, obstacle points {obstacle.sum()}, "
          f"u_x {fields['u_x'][~obstacle].min():.4g} .. {fields['u_x'][~obstacle].max():.4g}, "
          f"p {fields['p'][~obstacle].min():.4g} .. {fields['p'][~obstacle].max():.4g}")


if __name__ == "__main__":
    main()
