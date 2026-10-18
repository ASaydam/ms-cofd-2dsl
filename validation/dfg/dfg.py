"""
DFG 2D-1 benchmark (Schäfer & Turek 1996; docs/validation.md): steady laminar flow past a cylinder in a channel,
Re = 20, solved with the solver settings of the production (case template second-order: SIMPLEC, linearUpwind,
residualControl 1e-6) and its gmsh settings on three meshes. Only the geometry, the viscosity and the inlet differ.

    python validation/dfg/dfg.py size <level>                         cell size of a mesh level (coarse, medium, fine)
    python validation/dfg/dfg.py setup <case folder> <cell size>      template files, mesh.geo, initial fields
    python validation/dfg/dfg.py inlet <case folder>                  the parabolic inlet profile (after meshing)
    python validation/dfg/dfg.py functions <case folder>              function objects of the benchmark quantities
    python validation/dfg/dfg.py compare <root> <output folder>       table, mesh study and reference intervals

validation/dfg/run_dfg.sh makes one mesh level in the container.

Problem: channel 2.2 x 0.41 m, cylinder D = 0.1 m centred at (0.2, 0.2); inlet u = 4 Um y (H - y) / H^2 with
Um = 0.3 m/s (mean 0.2 m/s), outlet p = 0, no slip on the walls and the cylinder, nu = 1e-3 m^2/s.

Quantities (rho = 1, per unit depth: the meshes are one cell of 0.01 m in z):
    cD, cL   2 F / (U_mean^2 D), forces on the cylinder (forces function object, as tools/openfoam_attributes.py)
    dp       p(0.15, 0.2) - p(0.25, 0.2): the pressure of the cylinder faces, interpolated linearly in the angle to
             the front and the rear point of the cylinder
    La       recirculation length: from the rear point (x = 0.25) to where u_x on the line y = 0.2 turns positive

The inlet is a fixedValue list: the parabola evaluated at the inlet face centres (0/C of writeCellCentres), the same
values an exprFixedValue condition would give.
"""

import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

TEMPLATE = REPO / 'openfoam' / 'case-templates' / 'second-order' / 'base'
LENGTH, HEIGHT, DEPTH = 2.2, 0.41, 0.01
CENTRE, DIAMETER = (0.2, 0.2), 0.1
U_MAX, U_MEAN, NU = 0.3, 0.2, 1e-3
CAP = 50000   # endTime: a run that reaches it did not converge
# uniform cell size of the mesh levels (refinement ratio 2): D/20, D/40, D/80
LEVELS = {'coarse': DIAMETER / 20, 'medium': DIAMETER / 40, 'fine': DIAMETER / 80}
# Schäfer & Turek (1996) intervals and the reference values of the benchmark
REFERENCE = {'cD': (5.57, 5.59, 5.57954), 'cL': (0.0104, 0.0110, 0.010619), 'dp': (0.1172, 0.1176, 0.11752),
             'La': (0.0842, 0.0852, 0.0847)}

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  DFG 2D-1 benchmark (validation/dfg/dfg.py)
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def geo_text(size):
    """the gmsh file: the settings of geometries/geo-base-files (geo_base_1.geo, the circle file), exact circle arcs"""
    def g(value):  # coordinates without floating-point noise (0.15, not 0.15000000000000002)
        return f"{round(value, 9):g}"

    cx, cy = CENTRE
    r = DIAMETER / 2
    return f"""// DFG 2D-1 (Schaefer & Turek 1996): channel {LENGTH} x {HEIGHT} m, cylinder D = {DIAMETER} m at ({cx}, {cy}).
// gmsh settings of the production (geometries/geo-base-files): quads by recombination, one cell layer in z.
Mesh.Algorithm = 8; // Delaunay for quads

size = {size};

//inner geometry: the cylinder (right, top, left and bottom point, centre)
Point(1) = {{{g(cx + r)}, {g(cy)}, 0, size}};
Point(2) = {{{g(cx)}, {g(cy + r)}, 0, size}};
Point(3) = {{{g(cx - r)}, {g(cy)}, 0, size}};
Point(4) = {{{g(cx)}, {g(cy - r)}, 0, size}};
Point(13) = {{{g(cx)}, {g(cy)}, 0, size}};

Circle(1) = {{1, 13, 2}};
Circle(2) = {{2, 13, 3}};
Circle(3) = {{3, 13, 4}};
Circle(4) = {{4, 13, 1}};

Curve Loop(1) = {{1, 2, 3, 4}};

//outer geometry
Point(5) = {{0, 0, 0, size}};
Point(6) = {{{LENGTH}, 0, 0, size}};
Point(7) = {{{LENGTH}, {HEIGHT}, 0, size}};
Point(8) = {{0, {HEIGHT}, 0, size}};

Line(5) = {{5, 6}};
Line(6) = {{6, 7}};
Line(7) = {{7, 8}};
Line(8) = {{8, 5}};

Curve Loop(2) = {{5, 6, 7, 8}};
Plane Surface(2) = {{2, 1}};

Recombine Surface {{2}};

//Now make 3D by extrusion.
out[] =
Extrude {{ 0,0,{DEPTH} }}
{{
	Surface{{2}};
	Layers{{1}};
	Recombine;
}};

Physical Surface("frontAndBack") = {{2, out[0]}};
Physical Surface("outlet") = {{out[3]}};
Physical Surface("fixedWalls") = {{out[2], out[4]}};
Physical Surface("cylinder") = {{out[6], out[7], out[8], out[9]}};
Physical Surface("inlet") = {{out[5]}};
Physical Volume(100) = {{out[1]}};
"""


def field_text(name):
    """0/U and 0/p: the conditions of the production template, with the cylinder as a wall of its own"""
    if name == 'U':
        cls, dimensions, internal = 'volVectorField', '[0 1 -1 0 0 0 0]', 'uniform (0 0 0)'
        conditions = {'inlet': 'type fixedValue;\n        value uniform (0 0 0); // the parabola: dfg.py inlet',
                      'outlet': 'type zeroGradient;', 'fixedWalls': 'type noSlip;', 'cylinder': 'type noSlip;',
                      'frontAndBack': 'type empty;'}
    else:
        cls, dimensions, internal = 'volScalarField', '[0 2 -2 0 0 0 0]', 'uniform 0'
        conditions = {'inlet': 'type zeroGradient;', 'outlet': 'type fixedValue;\n        value uniform 0;',
                      'fixedWalls': 'type zeroGradient;', 'cylinder': 'type zeroGradient;',
                      'frontAndBack': 'type empty;'}
    blocks = ''.join(f"    {patch}\n    {{\n        {condition}\n    }}\n" for patch, condition in conditions.items())
    return (HEADER.format(cls=cls, obj=name) + f"dimensions      {dimensions};\n\ninternalField   {internal};\n\n"
            f"boundaryField\n{{\n{blocks}}}\n")


CHANGE_DICTIONARY = HEADER.format(cls='dictionary', obj='changeDictionaryDict') + """boundary
{
    frontAndBack
    {
        type            empty;
        inGroups        1(empty);
    }
    fixedWalls
    {
        type            wall;
        inGroups        1(wall);
    }
    cylinder
    {
        type            wall;
        inGroups        1(wall);
    }
}
"""


def setup(case, size):
    """a case folder: the solver files of the production template (fvSchemes and fvSolution unchanged, controlDict
    with a higher cap), nu, the DFG fields and patches, and mesh.geo"""
    case = Path(case)
    if case.exists():
        sys.exit(f"{case} exists")
    for part in ('system/fvSchemes', 'system/fvSolution', 'system/controlDict', 'constant/turbulenceProperties',
                 'foam.foam'):
        (case / part).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEMPLATE / part, case / part)
    control = case / 'system' / 'controlDict'
    text, count = re.subn(r"^endTime\s+\S+;.*$", f"endTime         {CAP};  // cap of the DFG benchmark",
                          control.read_text(), flags=re.MULTILINE)
    assert count == 1
    control.write_text(text, newline='\n')
    transport = (TEMPLATE / 'constant' / 'transportProperties').read_text()
    text, count = re.subn(r"^nu\s+\S+;", f"nu              {NU:.1e};", transport, flags=re.MULTILINE)
    assert count == 1
    (case / 'constant' / 'transportProperties').write_text(text, newline='\n')
    (case / 'system' / 'changeDictionaryDict').write_text(CHANGE_DICTIONARY, newline='\n')
    (case / '0').mkdir()
    for name in ('U', 'p'):
        (case / '0' / name).write_text(field_text(name), newline='\n')
    (case / 'mesh.geo').write_text(geo_text(size), newline='\n')


def inlet_centres(c_text):
    """face centres of the inlet patch from 0/C (writeCellCentres): list of (x, y, z)"""
    block = re.search(r"\n\s*inlet\s*\{(.*?)\n\s*\}", c_text, flags=re.DOTALL).group(1)
    match = re.search(r"value\s+nonuniform\s+List<vector>\s*(\d+)\s*\((.*?)\)\s*;", block, flags=re.DOTALL)
    count = int(match.group(1))
    numbers = [float(v) for v in match.group(2).replace('(', ' ').replace(')', ' ').split()]
    centres = [tuple(numbers[i:i + 3]) for i in range(0, len(numbers), 3)]
    assert len(centres) == count, (len(centres), count)
    return centres


def parabola(y):
    return 4 * U_MAX * y * (HEIGHT - y) / HEIGHT ** 2


def set_inlet(u_text, centres):
    """0/U with the parabola at the inlet face centres as a fixedValue list"""
    values = '\n'.join(f"({parabola(y):.10g} 0 0)" for _, y, _ in centres)
    entry = (f"    inlet\n    {{\n        type            fixedValue;\n"
             f"        value           nonuniform List<vector> {len(centres)}\n(\n{values}\n)\n;\n    }}")
    text, count = re.subn(r"    inlet\s*\{.*?\n    \}", lambda _: entry, u_text, count=1, flags=re.DOTALL)
    assert count == 1
    return text


def inlet(case):
    case = Path(case)
    centres = inlet_centres((case / '0' / 'C').read_text())
    (case / '0' / 'U').write_text(set_inlet((case / '0' / 'U').read_text(), centres), newline='\n')
    print(f"inlet: {len(centres)} faces, largest u {max(parabola(y) for _, y, _ in centres):.5f} m/s")


FUNCTIONS = """
functions
{
    // forces on the cylinder, rho = 1 (kinematic pressure), as tools/openfoam_attributes.py
    cylinderForces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0.2 0.2 0);
    }
    // p on the cylinder faces (face values, no interpolation), for dp
    cylinderSurface
    {
        type            surfaces;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   1;
        surfaceFormat   raw;
        interpolationScheme cell;
        fields          (p);
        surfaces
        (
            cylinder { type patch; patches (cylinder); interpolate false; }
        );
    }
    // U on the line y = 0.2 behind the cylinder, for La
    wakeLine
    {
        type            sets;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   1;
        setFormat       raw;
        interpolationScheme cellPoint;
        fields          (U);
        sets
        (
            centreline { type uniform; axis x; start (0.25 0.2 0.005); end (0.45 0.2 0.005); nPoints 8001; }
        );
    }
}
"""


def functions(case):
    control = Path(case) / 'system' / 'controlDict'
    text = control.read_text()
    if 'cylinderForces' not in text:
        control.write_text(text + FUNCTIONS, newline='\n')


# ------------------------------------------------------------------------------------------------------- results

def read_rows(path):
    return [[float(v) for v in line.split()] for line in open(path) if line.strip() and not line.startswith('#')]


def force_coefficients(line_values):
    """cD and cL from the last line of force.dat (time, total x y z, ...), per unit depth"""
    fx, fy = line_values[1] / DEPTH, line_values[2] / DEPTH
    scale = 0.5 * U_MEAN ** 2 * DIAMETER
    return fx / scale, fy / scale


def pressure_at(rows, angle):
    """p on the cylinder at an angle (radians, from the +x axis), linear in the angle between the two nearest faces;
    rows: x y z p of the faces"""
    cx, cy = CENTRE
    samples = sorted(((math.atan2(y - cy, x - cx) - angle + math.pi) % (2 * math.pi) - math.pi, p)
                     for x, y, _, p in rows)
    below = max((s for s in samples if s[0] <= 0), key=lambda s: s[0])
    above = min((s for s in samples if s[0] > 0), key=lambda s: s[0])
    return below[1] + (above[1] - below[1]) * (0 - below[0]) / (above[0] - below[0])


def recirculation_length(x, ux, rear=CENTRE[0] + DIAMETER / 2):
    """from the rear point to the first point where u_x on the centreline turns from negative to positive"""
    first = next((i for i, u in enumerate(ux) if u < 0), None)
    if first is None:
        return 0.0
    end = next((i for i in range(first, len(ux)) if ux[i] >= 0), None)
    if end is None:
        return float('nan')
    x0, x1, u0, u1 = x[end - 1], x[end], ux[end - 1], ux[end]
    return x0 + (x1 - x0) * (-u0) / (u1 - u0) - rear


def latest(folder):
    return max(Path(folder).iterdir(), key=lambda p: float(p.name))


def collect(case):
    """the benchmark quantities, iterations, residuals and cells of a solved case"""
    from extraction.check_runs import check_case

    case = Path(case)
    status = check_case(case)
    post = case / 'postProcessing'
    forces = next(latest(post / 'cylinderForces').glob('force*.dat'))
    cd, cl = force_coefficients(read_rows(forces)[-1])
    surface = read_rows(next(latest(post / 'cylinderSurface').glob('p_*.raw')))
    dp = pressure_at(surface, math.pi) - pressure_at(surface, 0.0)
    line = read_rows(next(latest(post / 'wakeLine').glob('centreline_U.xy')))
    la = recirculation_length([r[0] for r in line], [r[1] for r in line])
    cells = re.search(r"^\s*cells:\s+(\d+)", (case / 'log.checkMesh').read_text(), flags=re.MULTILINE)
    variant = case / 'mesh_variant'
    return {'status': status.get('status'), 'iterations': status.get('last_iteration'),
            'residual_Ux': status.get('residual_Ux'), 'residual_Uy': status.get('residual_Uy'),
            'residual_p': status.get('residual_p'), 'cells': int(cells.group(1)),
            'mesh_variant': variant.read_text().strip() if variant.exists() else '',
            'execution_time_s': status.get('execution_time'), 'cD': cd, 'cL': cl, 'dp': dp, 'La': la}


def inside(name, value):
    low, high, _ = REFERENCE[name]
    return value is not None and not math.isnan(value) and low <= value <= high


def compare(root, output):
    from tools.mesh_study import study

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {level: {'level': level, 'size': size, **collect(root / level)} for level, size in LEVELS.items()}
    fine, medium, coarse = results['fine'], results['medium'], results['coarse']
    studies, passed = {}, True
    for name, (low, high, reference) in REFERENCE.items():
        row = study(fine[name], medium[name], coarse[name], fine['cells'], medium['cells'], coarse['cells'])
        row = {key.replace('16x', 'fine').replace('4x', 'medium').replace('base', 'coarse'): value
               for key, value in row.items()}
        extrapolated = row.get('extrapolated')
        row.update({'quantity': name, 'reference': reference, 'low': low, 'high': high, 'fine': fine[name],
                    'fine_inside': inside(name, fine[name]),
                    'extrapolated_inside': extrapolated is not None and inside(name, extrapolated)})
        row['pass'] = row['fine_inside'] or row['extrapolated_inside']
        passed &= row['pass']
        studies[name] = row

    with open(output / 'dfg_meshes.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(fine), lineterminator='\n')
        writer.writeheader()
        writer.writerows(results.values())
    with open(output / 'dfg_study.csv', 'w', newline='') as file:
        fields = list(dict.fromkeys(k for row in studies.values() for k in row))
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(studies.values())
    (output / 'dfg_summary.json').write_text(json.dumps({'meshes': results, 'study': studies, 'pass': passed},
                                                        indent=1, default=str) + '\n')

    print(f"{'mesh':8s} {'size m':>9s} {'cells':>8s} {'iterations':>10s} {'res Ux':>9s} {'res Uy':>9s} "
          f"{'res p':>9s} {'cD':>9s} {'cL':>9s} {'dp':>9s} {'La':>9s}")
    for row in results.values():
        print(f"{row['level']:8s} {row['size']:9.6f} {row['cells']:8d} {row['iterations']!s:>10s} "
              f"{float(row['residual_Ux']):9.2e} {float(row['residual_Uy']):9.2e} {float(row['residual_p']):9.2e} "
              f"{row['cD']:9.5f} {row['cL']:9.6f} {row['dp']:9.5f} {row['La']:9.5f}")
    print(f"\n{'quantity':8s} {'reference':>10s} {'interval':>17s} {'fine':>9s} {'extrapol.':>9s} {'order':>6s} "
          f"{'GCI fine %':>10s} {'convergence':>12s} result")
    for name, row in studies.items():
        extrapolated = row.get('extrapolated')
        print(f"{name:8s} {row['reference']:10.6f} [{row['low']:.4f}, {row['high']:.4f}] {row['fine']:9.5f} "
              f"{extrapolated if extrapolated is None else round(extrapolated, 5)!s:>9s} "
              f"{row.get('order', float('nan')):6.2f} {row.get('gci_fine_%', float('nan')):10.3f} "
              f"{row['convergence']:>12s} {'PASS' if row['pass'] else 'OUTSIDE'} "
              f"(fine {'in' if row['fine_inside'] else 'out'}, extrapolated {'in' if row['extrapolated_inside'] else 'out'})")
    print(f"\n{'all quantities inside their interval' if passed else 'not all quantities inside'}; written to {output}")
    return passed


def main():
    command, args = sys.argv[1], sys.argv[2:]
    if command == 'size':
        print(LEVELS[args[0]])
    elif command == 'setup':
        setup(args[0], float(args[1]))
    elif command == 'inlet':
        inlet(args[0])
    elif command == 'functions':
        functions(args[0])
    elif command == 'compare':
        sys.exit(0 if compare(args[0], args[1]) else 1)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
