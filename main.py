"""
Command line entry of the MS-COFD-2DSL (Multi-Scale Central Object Flow Dataset - 2D Steady Laminar) pipeline.

The geometry stage writes the .geo files of all cases and the manifest; it is the first step of every production:

    python main.py geometry                      # all 4500 cases, all resolutions
    python main.py geometry --sizes smaller --geometries circle --start 1 --end 10

The production runs every case task by task (mesh, solver, arrays, attributes, archive), see hpc/run_task.py and
hpc/topup.py. For trying a few cases by hand, the meshes and OpenFOAM case folders can also be made in bulk:

    python main.py mesh --resolutions base --workers 4      # .msh files with gmsh
    python main.py foam-setup --resolutions base            # OpenFOAM case folders (openfoam/scripts/run_case.sh)
    python main.py all                                      # geometry, mesh and foam-setup

Settings are in config/dataset.yaml; the output goes to the data root (MSCOFD_DATA_ROOT, default outputs/).
"""

import argparse

from utils import REPO_ROOT, load_config, update_manifest
from geometries.write_geo import create_geo
from meshing.create_mesh import create_mesh
from openfoam.create_foam import create_foam


def run_geometry(config, args):
    rows = create_geo(config, args.sizes, args.geometries, args.resolutions, start=args.start, end=args.end)
    update_manifest(rows, config['paths']['manifest'])


def run_mesh(config, args):
    meshing = config['meshing']
    for resolution in args.resolutions:
        create_mesh(config['paths']['geo_files'] / resolution, config['paths']['mesh_files'] / resolution,
                    gmsh=meshing['gmsh'], threads=args.threads or meshing['threads'],
                    workers=args.workers or meshing['workers'], timeout=meshing['timeout'], overwrite=args.overwrite)


def run_foam_setup(config, args):
    for resolution in args.resolutions:
        # e.g. --init-from base: every 16x case starts from the solution of its base case
        init_from = None
        if args.init_from and args.init_from != resolution:
            init_from = config['paths']['cases'] / args.init_from
        create_foam(config['paths']['mesh_files'] / resolution, config['paths']['cases'] / resolution, resolution,
                    template_set=args.templates, init_from=init_from, overwrite=args.overwrite)


def main():
    # the config file is read first, the defaults of the other options come from it
    config_option = argparse.ArgumentParser(add_help=False)
    config_option.add_argument("--config", default=str(REPO_ROOT / "config" / "dataset.yaml"),
                               help="configuration file")
    config = load_config(config_option.parse_known_args()[0].config)

    parser = argparse.ArgumentParser(description="MS-COFD-2DSL dataset pipeline, settings are in config/dataset.yaml",
                                     parents=[config_option])
    stages = parser.add_subparsers(dest="stage", required=True)

    # options shared by geometry / mesh / foam-setup / all
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--resolutions", nargs="+", default=config['default_resolutions'],
                        choices=list(config['resolutions']), help="resolutions to work on")
    common.add_argument("--overwrite", action="store_true", help="create again the files that already exist")

    geometry = argparse.ArgumentParser(add_help=False)
    geometry.add_argument("--sizes", nargs="+", default=list(config['cases_per_geometry']),
                          choices=list(config['cases_per_geometry']))
    geometry.add_argument("--geometries", nargs="+", default=config['geometries'], choices=config['geometries'])
    geometry.add_argument("--start", type=int, default=1, help="first case number")
    geometry.add_argument("--end", type=int, default=None,
                          help="last case number (default: cases_per_geometry of config/dataset.yaml)")

    mesh = argparse.ArgumentParser(add_help=False)
    mesh.add_argument("--workers", type=int, default=None, help="meshes created at the same time")
    mesh.add_argument("--threads", type=int, default=None, help="gmsh threads per mesh")

    foam = argparse.ArgumentParser(add_help=False)
    foam.add_argument("--templates", default="second-order", choices=["first-order", "second-order"],
                      help="OpenFOAM settings, see openfoam/create_foam.py")
    foam.add_argument("--init-from", default=None, choices=list(config['resolutions']),
                      help="start the cases from the solution of this resolution's cases (e.g. 16x from base)")

    stages.add_parser("geometry", parents=[common, geometry], help="create .geo files and the manifest")
    stages.add_parser("mesh", parents=[common, mesh], help="create .msh files with gmsh")
    stages.add_parser("foam-setup", parents=[common, foam], help="create OpenFOAM case folders")
    stages.add_parser("all", parents=[common, geometry, mesh, foam], help="geometry, mesh and foam-setup")

    args = parser.parse_args()

    if args.stage in ("geometry", "all"):
        run_geometry(config, args)
    if args.stage in ("mesh", "all"):
        run_mesh(config, args)
    if args.stage in ("foam-setup", "all"):
        run_foam_setup(config, args)


if __name__ == "__main__":
    main()
