"""
Runs one task of the production: one run (base_fo, base_lu, 4x or 16x) of one case. Runs inside the container,
started by hpc/job_task.sh:

    python -m hpc.run_task <run> <case>        (MSCOFD_DATA_ROOT is the data root)

Steps, every one is skipped when its result exists, so an interrupted task (preempted, time limit) continues where
it stopped:

1. case folder: the template of the run (config/dataset.yaml, runs) and the .geo file of the resolution
2. mesh: gmsh -> gmshToFoam -> changeDictionary (patch types) -> checkMesh
3. initial fields (4x, 16x): mapFields from the base_lu solution of the case (<data root>/seeds/<case>)
4. simpleFoam until residualControl stops it, or the endTime cap; continued from the latest time step when the
   task was interrupted. Peak memory measured with /usr/bin/time -v.
5. base_lu: the solution is kept in seeds/<case> until the 4x and 16x runs of the case are done
6. arrays: resampled with VTK to the grid of the resolution -> results/<run>/<case>.npz (float32, 6 channels)
7. attributes with OpenFOAM's function objects -> results/<run>/<case>_attributes.json, and the pressure and wall
   shear along the obstacle surface -> results/<run>/<case>_surface.csv
8. archive: the case folder and the attribute work folder -> archives/<run>/<case>.tar.zst; uploaded with rclone
   (MSCOFD_REMOTE) and deleted after the upload is checked, see hpc/upload.py
9. provenance -> results/<run>/<case>_provenance.json: status, iterations, residuals, timings, peak memory, host,
   CPU, versions, code revision and container image

The state of the task (hpc/ledger.py) is set to done at the end, or to failed with the error.
"""

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import REPO_ROOT, load_config, read_csv  # noqa: E402
from extraction.build_arrays import build_sample  # noqa: E402
from extraction.check_runs import check_case  # noqa: E402
from extraction.resample_vtk import latest_time, resample_case  # noqa: E402
from hpc import ledger  # noqa: E402
from hpc.upload import upload_archive  # noqa: E402

TEMPLATES = REPO_ROOT / 'openfoam' / 'case-templates'
ATTRIBUTES_SCRIPT = REPO_ROOT / 'openfoam' / 'scripts' / 'openfoam_attributes.sh'

# one line per finished task in results/<run>/summary.csv (read by hpc/status.py and the cost study)
SUMMARY_FIELDS = ['case', 'status', 'iterations', 'execution_time_s', 'clock_time_s', 'peak_memory_kb', 'cells',
                  'host', 'cpu_model', 'node_cpus_allocated', 'slurm_job_id', 'finished']


class Task:
    def __init__(self, run, case):
        self.config = load_config()
        self.run, self.case = run, case
        self.settings = self.config['runs'][run]
        self.resolution = self.settings['resolution']
        self.root = Path(os.environ['MSCOFD_DATA_ROOT'])
        paths = self.config['paths']
        self.folder = paths['cases'] / run / case
        self.attributes_folder = paths['cases'] / run / f"{case}_attributes"
        self.seed = self.root / 'seeds' / case
        self.results = paths['results'] / run
        self.archive = paths['archives'] / run / f"{case}.tar.zst"
        self.timings = {}

    # ------------------------------------------------------------------------------------------------- helpers

    def shell(self, command, log, cwd=None, check=True, append=False):
        """runs a command, its output goes to log (in the case folder)"""
        with open((cwd or self.folder) / log, 'a' if append else 'w') as file:
            result = subprocess.run(command, cwd=cwd or self.folder, stdout=file, stderr=subprocess.STDOUT)
        if check and result.returncode != 0:
            raise RuntimeError(f"{' '.join(map(str, command))} failed with exit code {result.returncode}, see {log}")
        return result.returncode

    def timed(self, name, function, *args, **kwargs):
        start = time.time()
        value = function(*args, **kwargs)
        self.timings[name] = round(self.timings.get(name, 0) + time.time() - start, 2)
        return value

    # --------------------------------------------------------------------------------------------------- steps

    def mesh_ok(self):
        log = self.folder / 'log.checkMesh'
        return log.exists() and "Mesh OK." in log.read_text(errors='replace')

    def create_case(self):
        """
        steps 1 and 2: a new case folder with the converted mesh (a folder without a finished mesh is redone). A mesh
        that fails checkMesh is made again with the next of meshing.variants (gmsh options appended to the .geo
        file); the number and options of the variant used are written to mesh_variant, the checkMesh logs of the
        failed variants are kept as log.checkMesh.variant<number>
        """
        marker = self.folder / 'mesh_variant'
        if marker.exists():
            return
        if (self.folder / 'constant' / 'polyMesh' / 'owner').exists() and self.mesh_ok():
            marker.write_text("0\n")  # meshed before the variants were recorded
            return

        geo = (self.config['paths']['geo_files'] / self.resolution / f"{self.case}.geo").read_text()
        meshing = self.config['meshing']
        failed_logs = {}
        for number, options in enumerate(meshing['variants']):
            shutil.rmtree(self.folder, ignore_errors=True)
            shutil.copytree(TEMPLATES / self.settings['templates'] / self.resolution, self.folder)
            (self.folder / 'mesh.geo').write_text(geo + (f"\n// mesh variant {number}\n{options}\n" if options else ''))
            for name, text in failed_logs.items():
                (self.folder / name).write_text(text)

            self.timed('gmsh', self.shell, ['gmsh', '-3', 'mesh.geo', '-o', 'mesh.msh', '-format', 'msh2',
                                            '-nt', str(meshing['threads'])], 'log.gmsh')
            self.timed('gmshToFoam', self.shell, ['gmshToFoam', 'mesh.msh'], 'log.gmshToFoam')
            self.shell(['changeDictionary'], 'log.changeDictionary')
            self.shell(['checkMesh'], 'log.checkMesh', check=False)
            if self.mesh_ok():
                marker.write_text(f"{number} {options}".strip() + "\n")
                return
            failed_logs[f"log.checkMesh.variant{number}"] = (self.folder / 'log.checkMesh').read_text(errors='replace')
        raise RuntimeError(f"checkMesh failed with all {len(meshing['variants'])} mesh variants, see log.checkMesh")

    def map_fields(self):
        """step 3: initial fields from the base_lu solution of the case"""
        source = self.settings.get('init_from')
        if not source or (self.folder / 'mapped').exists():
            return
        if not (self.seed / 'system' / 'controlDict').exists():
            raise RuntimeError(f"no {source} solution of {self.case} in {self.seed}")
        self.timed('mapFields', self.shell, ['mapFields', str(self.seed), '-consistent', '-sourceTime', 'latestTime',
                                             '-mapMethod', 'cellPointInterpolate'], 'log.mapFields')
        (self.folder / 'mapped').write_text(f"{self.seed}\n")

    def solve(self):
        """step 4: simpleFoam, continued from the latest time step when the case was started before"""
        status = check_case(self.folder)
        if status['status'] in ('converged', 'end_time'):
            return status
        continued = (self.folder / 'log.simpleFoam').exists()
        if continued:
            self.shell(['foamDictionary', 'system/controlDict', '-entry', 'startFrom', '-set', 'latestTime'],
                       'log.foamDictionary')
        # /usr/bin/time -v adds the peak memory (Maximum resident set size) of every run to log.time
        self.timed('simpleFoam', self.shell, ['/usr/bin/time', '-v', '-a', '-o', 'log.time', 'simpleFoam'],
                   'log.simpleFoam', append=continued)
        return check_case(self.folder)

    def keep_seed(self):
        """step 5: the base_lu solution for the 4x and 16x runs (mesh, settings and the latest time step)"""
        if self.run != 'base_lu':
            return
        temporary = self.seed.with_name(f".{self.case}.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        shutil.copytree(self.folder / 'constant', temporary / 'constant')
        shutil.copytree(self.folder / 'system', temporary / 'system')
        last = latest_time(self.folder)
        shutil.copytree(self.folder / time_folder_name(self.folder, last), temporary / time_folder_name(self.folder, last))
        shutil.rmtree(self.seed, ignore_errors=True)
        os.replace(temporary, self.seed)

    def arrays(self):
        """step 6: the 6 channel sample of the run, float32"""
        nx, ny = self.config['extraction'][self.resolution]
        fields = self.timed('resample', resample_case, self.folder, nx, ny)
        sample = build_sample(fields['u_x'], fields['u_y'], fields['p'], fields['valid'])
        self.results.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.results / f"{self.case}.npz", sample=sample.astype(np.float32),
                            time=fields['time'])
        return fields['time']

    def attributes(self):
        """step 7: attributes with OpenFOAM in a copy of the case (hpc / tools/openfoam_attributes.py)"""
        from tools.openfoam_attributes import collect_case, obstacle_geometry

        row = next(row for row in read_csv(self.config['paths']['manifest']) if row['case_name'] == self.case)
        box, x_centroid, x_wake = obstacle_geometry(row, self.config['domain'])
        domain = [self.config['domain']['length'] / 1000, self.config['domain']['height'] / 1000]
        command = (['bash', str(ATTRIBUTES_SCRIPT), str(self.folder), str(self.attributes_folder)] +
                   [str(value) for value in box + [x_centroid, x_wake] + domain])
        with open(self.folder / 'log.attributes', 'w') as log:
            result = self.timed('attributes', subprocess.run, command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"openfoam_attributes.sh failed with exit code {result.returncode}, see log.attributes")
        values = collect_case(str(self.attributes_folder))
        with open(self.results / f"{self.case}_attributes.json", 'w') as file:
            json.dump(values, file, indent=1, default=float)
        # pressure and wall shear along the obstacle surface (x, y in mm, written by collect_case)
        shutil.copy(self.attributes_folder / 'surface.csv', self.results / f"{self.case}_surface.csv")
        return values

    def make_archive(self):
        """step 8 (first part): the case and the attribute work folder in one zstd archive"""
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.archive.with_name(self.archive.name + '.part')
        members = [self.case] + ([f"{self.case}_attributes"] if self.attributes_folder.exists() else [])
        command = f"tar -cf - -C '{self.folder.parent}' {' '.join(members)} | zstd -q -T1 -9 -o '{temporary}' -f"
        self.timed('archive', subprocess.run, ['bash', '-o', 'pipefail', '-c', command], check=True)
        if not temporary.exists():
            raise RuntimeError("archive was not written")
        os.replace(temporary, self.archive)

    def provenance(self, status, exported_time):
        """step 9: everything needed to describe and reproduce the run"""
        peaks = [int(line.split(':')[1]) for line in open(self.folder / 'log.time')
                 if 'Maximum resident set size' in line] if (self.folder / 'log.time').exists() else []
        versions = Path(os.environ.get('MSCOFD_VERSIONS', '/opt/mscofd/versions')) / 'summary.txt'
        cpu = next((line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo') if line.startswith('model name')), '')
        cells = next((line.split(':')[1].strip() for line in open(self.folder / 'log.checkMesh')
                      if line.strip().startswith('cells:')), '') if (self.folder / 'log.checkMesh').exists() else ''
        record = {
            'run': self.run, 'case': self.case, 'resolution': self.resolution, 'templates': self.settings['templates'],
            'status': status['status'], 'iterations': status.get('last_iteration'),
            'solver_runs': status.get('runs'), 'execution_time_s': status.get('execution_time'),
            'clock_time_s': status.get('clock_time'), 'residual_Ux': status.get('residual_Ux'),
            'residual_Uy': status.get('residual_Uy'), 'residual_p': status.get('residual_p'),
            'mesh_ok': status.get('mesh_ok'), 'cells': cells, 'exported_time': exported_time,
            'mesh_variant': (self.folder / 'mesh_variant').read_text().strip()
            if (self.folder / 'mesh_variant').exists() else '',
            'peak_memory_kb': max(peaks) if peaks else None, 'timings_s': self.timings,
            'host': platform.node().split('.')[0], 'cpu_model': cpu,
            'node_cpus_allocated': os.environ.get('MSCOFD_NODE_ALLOCATED', ''),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
            'code_revision': (REPO_ROOT / 'REVISION').read_text().strip() if (REPO_ROOT / 'REVISION').exists() else '',
            'container_image': os.environ.get('MSCOFD_IMAGE', ''),
            'container_digest': os.environ.get('MSCOFD_IMAGE_DIGEST', ''),
            'versions': versions.read_text() if versions.exists() else '',
            'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        }
        (self.folder / 'provenance.json').write_text(json.dumps(record, indent=1, default=str))
        shutil.copy(self.folder / 'provenance.json', self.results / f"{self.case}_provenance.json")

        # many jobs on different nodes finish at the same time: appended under a lock that holds across nodes
        with ledger.DirectoryLock(self.results / '.summary.lock'), \
                open(self.results / 'summary.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction='ignore', lineterminator='\n')
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(record)
        return record

    def release_seed(self):
        """the base_lu solution is removed when the 4x and 16x runs of the case are done"""
        if not self.settings.get('init_from'):
            return
        (self.seed / f"done_{self.run}").touch()
        users = [run for run, settings in self.config['runs'].items() if settings.get('init_from') == 'base_lu']
        if all((self.seed / f"done_{run}").exists() for run in users):
            shutil.rmtree(self.seed, ignore_errors=True)

    # ------------------------------------------------------------------------------------------------- the task

    def execute(self):
        self.timed('setup', self.create_case)
        self.map_fields()
        status = self.solve()
        if status['status'] not in ('converged', 'end_time'):
            raise RuntimeError(f"solver stopped without converging or reaching endTime ({status['status']})")

        self.keep_seed()
        exported_time = self.arrays()
        self.attributes()
        record = self.provenance(status, exported_time)
        self.make_archive()
        upload_archive(self.archive, self.run, self.case)

        shutil.rmtree(self.attributes_folder, ignore_errors=True)
        shutil.rmtree(self.folder, ignore_errors=True)
        self.release_seed()
        return record


def time_folder_name(case_folder, value):
    """name of the time folder with the given value (OpenFOAM writes 3880 or 3880.5)"""
    for path in Path(case_folder).iterdir():
        try:
            if path.is_dir() and float(path.name) == value:
                return path.name
        except ValueError:
            continue
    raise FileNotFoundError(f"no time folder {value} in {case_folder}")


def main():
    run, case = sys.argv[1], sys.argv[2]
    try:
        record = Task(run, case).execute()
        ledger.set_state(run, case, 'done', f"{record['status']} {record['iterations']} {os.environ.get('SLURM_JOB_ID', '')}")
        print(f"{run} {case}: {record['status']} after {record['iterations']} iterations")
    except Exception as error:
        traceback.print_exc()
        ledger.set_state(run, case, 'failed', f"{type(error).__name__}: {error}".replace('\n', ' ')[:500])
        sys.exit(1)


if __name__ == "__main__":
    main()
