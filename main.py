"""
Command line entry of the MS-COFD-2DSL (Multi-Scale Central Object Flow Dataset - 2D Steady Laminar) pipeline,
every stage reads the
output of the previous one from disk:

    python main.py geometry       # .geo files of all cases + outputs/manifest.csv
    python main.py mesh           # .msh files with gmsh
    python main.py foam-setup     # OpenFOAM case folders
    python main.py all            # the three stages above, one after the other

    (OpenFOAM runs on a cluster or workstation, see openfoam/scripts/)
    python -m extraction.check_runs outputs/cases/base
    python extraction/extract_fields.py outputs/cases/base outputs/extracts/base --dims 260 120   # ParaView env

    python main.py build-arrays base
    python main.py build-arrays 16x

Settings are in config/dataset.yaml, every stage has options to work on a part of the dataset, e.g.

    python main.py geometry --sizes smaller --geometries circle --start 1 --end 10
    python main.py mesh --resolutions base --workers 4
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
                    template_set=args.templates, init_from=init_from,
                    overwrite=args.overwrite)


def run_build_arrays(config, args):
    # imported here, the other stages do not need it
    from extraction.build_arrays import build_arrays

    nx, ny = config['extraction'][args.resolution]
    build_arrays(args.extract_dir or config['paths']['extracts'] / args.resolution,
                 args.output_dir or config['paths']['arrays'] / args.resolution, args.resolution, nx, ny,
                 dtype=args.dtype)


def main():
    # the config file is read first, the defaults of the other options come from it
    config_option = argparse.ArgumentParser(add_help=False)
    config_option.add_argument("--config", default=str(REPO_ROOT / "config" / "dataset.yaml"), help="configuration file")
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

    stages.add_parser("geometry", parents=[common, geometry], help="create .geo files and the manifest")
    stages.add_parser("mesh", parents=[common, mesh], help="create .msh files with gmsh")
    foam = argparse.ArgumentParser(add_help=False)
    foam.add_argument("--templates", default="second-order", choices=["first-order", "second-order"],
                      help="OpenFOAM settings, see openfoam/create_foam.py")
    foam.add_argument("--init-from", default=None, choices=list(config['resolutions']),
                      help="start the cases from the solution of this resolution's cases (e.g. 16x from base)")

    stages.add_parser("foam-setup", parents=[common, foam], help="create OpenFOAM case folders")
    stages.add_parser("all", parents=[common, geometry, mesh, foam], help="geometry, mesh and foam-setup")

    arrays = stages.add_parser("build-arrays", help="build the dataset arrays from the ParaView extracts")
    arrays.add_argument("resolution", choices=list(config['extraction']))
    arrays.add_argument("--extract-dir", default=None, help="default: paths.extracts/<resolution>")
    arrays.add_argument("--output-dir", default=None, help="default: paths.arrays/<resolution>")
    arrays.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                        help="float32 keeps every digit of the exported fields (5 significant digits)")


    args = parser.parse_args()

    if args.stage in ("geometry", "all"):
        run_geometry(config, args)
    if args.stage in ("mesh", "all"):
        run_mesh(config, args)
    if args.stage in ("foam-setup", "all"):
        run_foam_setup(config, args)
    if args.stage == "build-arrays":
        run_build_arrays(config, args)


if __name__ == "__main__":
    main()
