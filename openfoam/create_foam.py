"""
This module creates one OpenFOAM case folder for every .msh file.

A case is a copy of openfoam/case-templates/<template set>/<resolution> (0/, constant/, system/) with the mesh
copied in as mesh.msh. The mesh is converted and the solver is run by openfoam/scripts/run_case.sh.

Template sets (the runs of every case are listed in config/dataset.yaml):

    first-order   base only: first-order upwind convection (a cheaper, more diffusive low-fidelity solution)
    second-order  base, 4x and 16x: second-order upwind convection (linearUpwind)

Both sets run to initial residuals of 1e-6 (residualControl), endTime is only a cap.

A case can start from the solution of another case with the same geometry (e.g. 16x from base): its init_from
file holds the path of that case, and run_case.sh maps its latest time step onto the new mesh (mapFields).
"""

import shutil
from pathlib import Path

from utils import create_folder

TEMPLATE_FOLDER = Path(__file__).resolve().parent / 'case-templates'


def create_foam(mesh_directory, cases_directory, resolution, template_set='second-order', init_from=None,
                overwrite=False):
    """
    creates the OpenFOAM cases of one resolution
    :param mesh_directory: folder of the .msh files
    :param cases_directory: output folder, a sub-folder is created for every case
    :param resolution: base, 4x or 16x, selects the case template
    :param template_set: first-order (base only) or second-order (see above)
    :param init_from: folder of already set up cases with the same names (e.g. the base cases), every case starts
     from the latest time step of its namesake there; None starts from the uniform fields in 0/
    :param overwrite: delete and create again the cases that already exist
    :return: list of the created case folders
    """
    template = TEMPLATE_FOLDER / template_set / resolution
    if not template.exists():
        raise FileNotFoundError(f"No OpenFOAM case template for '{template_set}/{resolution}' in {TEMPLATE_FOLDER}")

    mesh_directory, cases_directory = Path(mesh_directory), Path(cases_directory)
    create_folder(cases_directory)

    created_cases = []
    for mesh_file in sorted(mesh_directory.glob("*.msh")):
        case_folder = cases_directory / mesh_file.stem

        if case_folder.exists():
            if not overwrite:
                continue
            shutil.rmtree(case_folder)

        # copy base case to the current case and copy the generated mesh file
        shutil.copytree(template, case_folder)
        shutil.copy(mesh_file, case_folder / "mesh.msh")

        if init_from is not None:
            source_case = Path(init_from).resolve() / mesh_file.stem
            if not source_case.exists():
                raise FileNotFoundError(f"{case_folder.name}: no case to start from in {init_from}")
            (case_folder / "init_from").write_text(f"{source_case}\n")

        created_cases.append(case_folder)

    print(f"Created {len(created_cases)} {resolution} cases ({template_set} settings) in {cases_directory}")
    return created_cases
