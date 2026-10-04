"""
Flow attributes of OpenFOAM cases (latest time step), computed with OpenFOAM's own tools on the simulation mesh.

    python -m tools.openfoam_attributes run cases       # with OpenFOAM loaded (standard library only)
    python -m tools.openfoam_attributes collect cases   # without OpenFOAM, needs numpy
    (--resolutions base 4x 16x for a mesh convergence study, see tools/mesh_study.py)

run:      for every case folder <cases>/<resolution>/<case>, openfoam/scripts/openfoam_attributes.sh runs in
          <cases>/../openfoam-attributes/<resolution>/<case> (the case itself is not changed). The obstacle box,
          centroid and downstream end come from the manifest (--manifest).
collect:  reads the results of every work folder and writes <cases>/../openfoam_attributes.csv, one row per case
          and resolution, and surface.csv (x, y in mm, p, wall shear stress) in every work folder.

Attributes (same names as the array based ones of tools/flow_attributes.py; mm, m/s, rho = 1, per unit depth
where integrated; the first 5 mm of the channel are left out where noted, the inlet corners are singular):

    Fx, Fy, Fx_pressure, Fy_pressure, Fx_viscous, Fy_viscous   forces on the obstacle (forces), 10 mm depth
    max_wall_shear, front_pressure, base_pressure              obstacle faces (wallShearStress, surfaces);
                                                               base pressure = mean p on the rear 10 % of its length
    mean_wall_shear            |wall shear stress| averaged over the obstacle faces, area weighted
                               (mag + surfaceFieldValue)
    dp_inlet_outlet            area average of p on the inlet minus the outlet (surfaceFieldValue)
    dp_sections                average p on the cross-section 10 mm behind the inlet minus the one 10 mm before the
                               outlet (sets), without the singular inlet corners
    peak_vorticity             largest |vorticity| of the cells, first 5 mm left out (vorticity)
    mean_vorticity             |vorticity| averaged over the cells (area weighted), first 5 mm left out
    dissipation                nu * sum of |grad U|^2 * cell area, first 5 mm left out (grad)
    recirculation_area_mm2     area of the cells with u_x < 0
    recirculation_length_mm    furthest cell with u_x < 0 behind the obstacle, from its downstream end
    flow_split_upper           share of the flow above the obstacle at its centroid (sets)
    wake_min_u_x               smallest u_x 20 mm behind the obstacle (sets)
    outlet_reverse_flow        flow rate with u_x < 0 in the last 5 mm of the channel, relative to the net flow rate
                               (cells, area weighted); above 0 reverse flow reaches the outlet, where the boundary
                               condition (zero gradient) then sets part of the solution
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "openfoam" / "scripts" / "openfoam_attributes.sh"
NU = 1e-4        # m^2/s
DEPTH = 0.01     # m, thickness of the 2D meshes

FIELDS = ['case_name', 'resolution', 'status', 'time', 'cells', 'faces', 'Fx', 'Fy', 'Fx_pressure', 'Fy_pressure',
          'Fx_viscous', 'Fy_viscous', 'max_wall_shear', 'mean_wall_shear', 'front_pressure', 'base_pressure',
          'dp_inlet_outlet', 'dp_sections', 'peak_vorticity', 'mean_vorticity', 'dissipation', 'recirculation_area_mm2', 'recirculation_length_mm',
          'flow_split_upper', 'wake_min_u_x', 'outlet_reverse_flow']

RE_VECTOR = re.compile(r"\(([-\d.eE+]+)\s+([-\d.eE+]+)\s+[-\d.eE+]+\)")


# ---------------------------------------------------------------------------------------------------- run (OpenFOAM)

def obstacle_geometry(row, domain, margin=10.0):
    """
    box around the obstacle (bounding box + margin, kept 1 mm away from the channel walls and ends), x of its
    centroid and x 20 mm behind it, all in m
    :param domain: domain block of config/dataset.yaml (length and height in mm)
    """
    length, height = domain['length'], domain['height']
    xs = [float(row[k]) for k in ('x1', 'x2', 'x3', 'x4', 'z11', 'z21', 'z31', 'z41') if row.get(k)]
    ys = [float(row[k]) for k in ('y1', 'y2', 'y3', 'y4', 'z12', 'z22', 'z32', 'z42') if row.get(k)]
    box = [max(min(xs) - margin, 1) / 1000, max(min(ys) - margin, 1) / 1000,
           min(max(xs) + margin, length - 1) / 1000, min(max(ys) + margin, height - 1) / 1000]
    return box, float(row['centroid_x']) / 1000, min(max(xs) + 20, length - 1) / 1000


def run(cases, manifest_path, resolutions, domain):
    with open(manifest_path, newline='') as file:
        manifest = {row['case_name']: row for row in csv.DictReader(file)}

    for resolution in resolutions:
        for case_folder in sorted(path for path in (cases / resolution).iterdir() if path.is_dir()):
            work = cases.parent / "openfoam-attributes" / resolution / case_folder.name
            box, x_centroid, x_wake = obstacle_geometry(manifest[case_folder.name], domain)
            size = [domain['length'] / 1000, domain['height'] / 1000]
            result = subprocess.run(["bash", str(SCRIPT), str(case_folder), str(work)] +
                                    [str(value) for value in box + [x_centroid, x_wake] + size],
                                    capture_output=True, text=True)
            status = "ok" if result.returncode == 0 else "failed"
            (work / "status").write_text(status + "\n")
            print(f"{case_folder.name:28s} {resolution:5s} {status}")


# ------------------------------------------------------------------------------------------------- collect (numpy)

def read_field(path):
    """internal field of an ascii OpenFOAM field file, as an (N,) or (N, components) array"""
    import numpy as np

    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("internalField"))
    count = int(lines[start + 1])
    values = " ".join(lines[start + 3:start + 3 + count]).replace("(", " ").replace(")", " ").split()
    array = np.array(values, dtype=float)
    return array if array.size == count else array.reshape(count, -1)


def read_raw(path):
    """rows of a raw surface or set file"""
    return [[float(v) for v in line.split()] for line in open(path) if line.strip() and not line.startswith('#')]


def last_values(pattern):
    """numbers of the last line of a function object .dat file"""
    lines = [line for line in open(glob.glob(pattern)[0]) if line.strip() and not line.startswith('#')]
    return lines[-1]


def collect_case(work):
    import numpy as np

    time = next(p for p in os.listdir(work) if re.fullmatch(r"[0-9.]+", p))
    row = {'time': time}

    # forces: total, pressure and viscous vectors, written as "(x y z)" groups by older OpenFOAM versions (v1706)
    # and as plain columns by newer ones (v2512: time total_x total_y total_z pressure_x ... viscous_z)
    line = last_values(os.path.join(work, "postProcessing", "obstacleForces", "*", "force*.dat"))
    vectors = [(float(x), float(y)) for x, y in RE_VECTOR.findall(line)]
    if len(vectors) < 3:
        values = [float(value) for value in line.split()[1:10]]
        vectors = [(values[0], values[1]), (values[3], values[4]), (values[6], values[7])]
    (row['Fx'], row['Fy']), (row['Fx_pressure'], row['Fy_pressure']), (row['Fx_viscous'], row['Fy_viscous']) = vectors[:3]

    # obstacle surface
    folder = glob.glob(os.path.join(work, "postProcessing", "obstacleSurface", "*"))[0]
    pressure = np.array(read_raw(glob.glob(os.path.join(folder, "p_*.raw"))[0]))
    shear = np.array(read_raw(glob.glob(os.path.join(folder, "wallShearStress_*.raw"))[0]))
    with open(os.path.join(work, "surface.csv"), 'w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['x_mm', 'y_mm', 'p', 'wall_shear_x', 'wall_shear_y'])
        writer.writerows(np.column_stack([pressure[:, 0] * 1000, pressure[:, 1] * 1000, pressure[:, 3],
                                          shear[:, 3], shear[:, 4]]).tolist())
    x_faces = pressure[:, 0]
    rear = x_faces >= x_faces.max() - 0.1 * (x_faces.max() - x_faces.min())
    row.update({'faces': len(pressure), 'max_wall_shear': float(np.hypot(shear[:, 3], shear[:, 4]).max()),
                'front_pressure': float(pressure[:, 3].max()), 'base_pressure': float(pressure[rear, 3].mean())})
    row['mean_wall_shear'] = float(last_values(os.path.join(work, "postProcessing", "obstacleShear", "*",
                                                            "*.dat")).split()[1])

    # pressure drop
    inlet = float(last_values(os.path.join(work, "postProcessing", "inletPressure", "*", "*.dat")).split()[1])
    outlet = float(last_values(os.path.join(work, "postProcessing", "outletPressure", "*", "*.dat")).split()[1])
    row['dp_inlet_outlet'] = inlet - outlet

    # cell fields
    centres = read_field(os.path.join(work, time, "C"))
    area = read_field(os.path.join(work, time, "V")) / DEPTH
    row['cells'] = len(area)
    ux = read_field(os.path.join(work, time, "U"))[:, 0]
    vorticity = read_field(os.path.join(work, time, "vorticity"))[:, 2]
    grad = read_field(os.path.join(work, time, "grad(U)"))
    x_mm = centres[:, 0] * 1000
    region = x_mm > 5
    in_plane = grad[:, [0, 1, 3, 4]]  # dUx/dx, dUy/dx, dUx/dy, dUy/dy
    row['peak_vorticity'] = float(np.abs(vorticity[region]).max())
    row['mean_vorticity'] = float((np.abs(vorticity[region]) * area[region]).sum() / area[region].sum())
    row['dissipation'] = float(NU * ((in_plane[region] ** 2).sum(axis=1) * area[region]).sum())
    reverse = ux < 0
    row['recirculation_area_mm2'] = float(area[reverse].sum() * 1e6)
    x_end = x_faces.max() * 1000
    behind = reverse & (x_mm > x_end)
    row['recirculation_length_mm'] = float(x_mm[behind].max() - x_end) if behind.any() else 0.0
    end = x_mm > x_mm.max() - 5
    row['outlet_reverse_flow'] = float((np.clip(-ux[end], 0, None) * area[end]).sum() / (ux[end] * area[end]).sum())

    # lines across the channel, sampled from each wall up to the obstacle
    lines = os.path.join(work, "postProcessing", "lines", "*")
    integrate = getattr(np, 'trapezoid', None) or np.trapz  # numpy 2 renamed trapz

    def read_line(name):
        """
        y, u_x and p of a sampled line: one file with the columns y p U_x U_y U_z when U and p are sampled together,
        <name>_U.xy (y U_x U_y U_z) and <name>_p.xy (y p) when they are written separately
        """
        combined = glob.glob(os.path.join(lines, f"{name}_p_U.xy"))
        if combined:
            line = np.array(read_raw(combined[0]))
            return line[:, 0], line[:, 2], line[:, 1]
        line = np.array(read_raw(glob.glob(os.path.join(lines, f"{name}_U.xy"))[0]))
        pressure = glob.glob(os.path.join(lines, f"{name}_p.xy"))
        return line[:, 0], line[:, 1], np.array(read_raw(pressure[0]))[:, 1] if pressure else None

    def flux_to_obstacle(name):
        """flow rate from the wall to the obstacle: the first contiguous part of the sampled line"""
        y, u, _ = read_line(name)
        steps = np.abs(np.diff(y))
        breaks = np.nonzero(steps > 1.5 * np.median(steps))[0]
        end = breaks[0] + 1 if len(breaks) else len(y)
        return abs(integrate(u[:end], y[:end]))

    below, above = flux_to_obstacle("centroidBelow"), flux_to_obstacle("centroidAbove")
    row['flow_split_upper'] = float(above / (above + below))
    row['wake_min_u_x'] = float(read_line("wakeLine")[1].min())

    # pressure drop between cross-sections 10 mm behind the inlet and 10 mm before the outlet (the area average on
    # the inlet patch includes the singular inlet corners, the cross-sections do not)
    def section_pressure(name):
        y, _, pressure = read_line(name)
        return integrate(pressure, y) / (y[-1] - y[0])

    row['dp_sections'] = float(section_pressure("inletSection") - section_pressure("outletSection"))
    return row


def collect(cases, resolutions):
    rows = []
    for resolution in resolutions:
        for work in sorted(path for path in (cases.parent / "openfoam-attributes" / resolution).iterdir()
                           if path.is_dir()):
            row = {'case_name': work.name, 'resolution': resolution}
            status_file = work / "status"
            row['status'] = status_file.read_text().strip() if status_file.exists() else 'not run'
            if row['status'] == 'ok':
                try:
                    row.update(collect_case(str(work)))
                except Exception as error:
                    row['status'] = f'collect failed: {error}'
            print(f"{work.name:28s} {resolution:5s} {row['status']}")
            rows.append(row)

    output = cases.parent / "openfoam_attributes.csv"
    with open(output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Attributes written to {output}")


def main():
    parser = argparse.ArgumentParser(description="Flow attributes computed with OpenFOAM")
    parser.add_argument("step", choices=["run", "collect"], help="run (with OpenFOAM) or collect (with numpy)")
    parser.add_argument("cases", help="folder with one sub-folder per resolution, e.g. cases")
    parser.add_argument("--manifest", default=None,
                        help="manifest with the obstacle coordinates of the cases (run), default: paths.manifest of "
                             "config/dataset.yaml")
    parser.add_argument("--domain", type=float, nargs=2, default=None,
                        help="domain length and height in mm (run), default: domain of config/dataset.yaml")
    parser.add_argument("--resolutions", nargs="+", default=["base", "16x"])
    args = parser.parse_args()

    if args.step == "run":
        # imported here, the run step reads the configuration only when a default is needed
        from utils import load_config
        config = load_config() if args.manifest is None or args.domain is None else None
        manifest = args.manifest or config['paths']['manifest']
        domain = ({'length': args.domain[0], 'height': args.domain[1]} if args.domain else config['domain'])
        run(Path(args.cases), manifest, args.resolutions, domain)
    else:
        collect(Path(args.cases), args.resolutions)


if __name__ == "__main__":
    main()
