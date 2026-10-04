"""
This module creates the .msh files from the .geo files with gmsh.

gmsh is called once for every .geo file (gmsh -3 <file> -format msh2 -nt <threads>).
A mesh is first written to a temporary file and renamed when gmsh finishes, so an interrupted run can be started
again: the finished meshes are skipped. Every run writes mesh_log.csv next to the mesh folders.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils import create_folder, read_csv, write_csv

# used when only the gmsh Python module is installed, with no launcher script to call; the user's gmsh option
# files (saved preferences of the GUI) are not read, they could change the mesh settings
BOOTSTRAP = "import sys, gmsh; gmsh.initialize(sys.argv, readConfigFiles=False, run=True); gmsh.finalize()"

# gmsh prints the size of the mesh on stdout, e.g. "Info    : 97580 nodes 207368 elements"
RE_COUNTS = re.compile(r"(\d+) nodes (\d+) elements")

LOG_FIELDS = ['case_name', 'resolution', 'status', 'seconds', 'nodes', 'elements', 'geo_file', 'msh_file', 'gmsh']


def resolve_gmsh_command(spec="gmsh"):
    """
    Return the command prefix that runs gmsh, as a list to prepend to its arguments.

    A native gmsh binary is used directly. The pip wheel ships no binary on Windows, only a
    `gmsh.bat` shim that CreateProcess cannot execute, so fall back to running the launcher
    script (or an inline bootstrap) under this interpreter instead.
    """
    spec = os.environ.get("GMSH_BIN", spec)
    found = shutil.which(spec) or (spec if Path(spec).exists() else None)

    if found:
        path = Path(found)
        if path.suffix.lower() in (".bat", ".cmd"):
            launcher = path.with_suffix("")
            if launcher.exists():
                return [sys.executable, str(launcher)]
        else:
            return [str(path)]

    try:
        import gmsh  # noqa: F401
    except ImportError:
        return None

    return [sys.executable, "-c", BOOTSTRAP]


def gmsh_version(command_prefix):
    try:
        result = subprocess.run(command_prefix + ["--version"], capture_output=True, text=True, timeout=60)
        output = ((result.stdout or "") + (result.stderr or "")).strip().splitlines()
        # gmsh may print warnings (e.g. about its option files) before the version
        versions = [line.strip() for line in output if re.fullmatch(r"\d+\.\d+\.\d+\S*", line.strip())]
        return versions[-1] if versions else (output[-1] if output else "unknown")
    except Exception:
        return "unknown"


def mesh_one(command_prefix, geo_file, msh_file, threads, timeout):
    """
    runs gmsh for a single .geo file
    :return: status (ok / failed / timeout), duration in seconds and the gmsh output
    """
    temporary_file = msh_file.with_name(msh_file.name + ".part")
    arguments = ["-3", str(geo_file), "-o", str(temporary_file), "-format", "msh2", "-nt", str(threads)]

    started = time.perf_counter()
    try:
        result = subprocess.run(command_prefix + arguments, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout or "") + (result.stderr or "")
        status = "ok" if result.returncode == 0 and temporary_file.exists() else "failed"
    except subprocess.TimeoutExpired:
        output, status = "", "timeout"
    seconds = round(time.perf_counter() - started, 3)

    # only a finished mesh gets its final name
    if status == "ok":
        os.replace(temporary_file, msh_file)
    elif temporary_file.exists():
        temporary_file.unlink()

    return status, seconds, output


def create_mesh(geo_directory, mesh_directory, gmsh="gmsh", threads=3, workers=1, timeout=3600, overwrite=False):
    """
    creates a .msh file for every .geo file of a folder
    :param geo_directory: folder of the .geo files (one resolution)
    :param mesh_directory: output folder of the .msh files
    :param gmsh: gmsh executable (GMSH_BIN environment variable overrides it)
    :param threads: threads used by gmsh for a single mesh
    :param workers: number of meshes created at the same time
    :param timeout: seconds before a single mesh is given up
    :param overwrite: create the meshes again even if they exist
    :return: log rows, one per .geo file
    """
    command_prefix = resolve_gmsh_command(gmsh)
    if command_prefix is None:
        raise RuntimeError("gmsh not found: install it (pip install gmsh) or set GMSH_BIN")
    version = gmsh_version(command_prefix)

    geo_directory, mesh_directory = Path(geo_directory), Path(mesh_directory)
    create_folder(mesh_directory)
    geo_files = sorted(geo_directory.glob("*.geo"))
    print(f"Meshing {len(geo_files)} files of {geo_directory} with gmsh {version}")

    # rows of the previous runs are kept for the meshes that are skipped
    log_file = mesh_directory.parent / f"mesh_log_{geo_directory.name}.csv"
    previous_rows = {row['case_name']: row for row in read_csv(log_file)} if log_file.exists() else {}

    def run(geo_file):
        msh_file = mesh_directory / f"{geo_file.stem}.msh"
        row = {'case_name': geo_file.stem, 'resolution': geo_directory.name, 'geo_file': str(geo_file),
               'msh_file': str(msh_file), 'gmsh': version}

        if msh_file.exists() and not overwrite:
            return previous_rows.get(geo_file.stem, {**row, 'status': 'skipped (exists)', 'gmsh': ''})

        status, seconds, output = mesh_one(command_prefix, geo_file, msh_file, threads, timeout)
        counts = RE_COUNTS.findall(output)
        row.update({'status': status, 'seconds': seconds,
                    'nodes': counts[-1][0] if counts else '', 'elements': counts[-1][1] if counts else ''})
        print(f"{status:>7} {seconds:8.1f} s  {msh_file.name}")
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        log_rows = list(executor.map(run, geo_files))

    failed = [row['case_name'] for row in log_rows if row['status'] in ('failed', 'timeout')]
    if failed:
        print(f"{len(failed)} meshes failed: {failed}")

    write_csv(log_rows, log_file, LOG_FIELDS)
    return log_rows
