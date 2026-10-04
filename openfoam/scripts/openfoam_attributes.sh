#!/bin/bash
#
# Computes flow attributes of the latest time step of a case with OpenFOAM's own tools. The obstacle is part of
# the fixedWalls patch of the meshes, so the case is copied to a work folder and the fixedWalls faces inside a box
# around the obstacle are moved to their own patch there (topoSet + createPatch); the case itself is not changed.
# OpenFOAM has to be loaded.
#
#   bash openfoam_attributes.sh <case folder> <work folder> <xmin> <ymin> <xmax> <ymax> <x centroid> <x wake>
#
# box, x centroid (obstacle centroid) and x wake (20 mm behind the obstacle) in m.
# Results in the work folder (read by tools/openfoam_attributes.py):
#   postProcessing/obstacleForces/<time>/force*.dat          total, pressure and viscous force (rho = 1)
#   postProcessing/obstacleSurface/<time>/*_obstacle.raw     p and wallShearStress on the obstacle faces
#   postProcessing/inletPressure, outletPressure/<time>/     area average of p on the inlet and the outlet
#   postProcessing/lines/<time>/*_U.xy                        U across the channel at x centroid and x wake
#   <time>/U, vorticity, grad(U), C, V                        cell values, cell centres and cell volumes

set -e
case_folder=$1; work=$2; xmin=$3; ymin=$4; xmax=$5; ymax=$6; x_centroid=$7; x_wake=$8
PYTHON=${PYTHON:-python3}

rm -rf "$work"; mkdir -p "$work"
latest=$(ls "$case_folder" | grep -E '^[0-9.]+$' | sort -g | tail -1)
cp -r "$case_folder/constant" "$case_folder/system" "$case_folder/$latest" "$work/"
cd "$work"
sed -i 's/^startFrom .*/startFrom       latestTime;/' system/controlDict

# the obstacle faces: fixedWalls faces inside the box (the channel walls are outside of it)
cat > system/topoSetDict <<EOT
FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions
(
    { name obstacleFaces; type faceSet; action new; source patchToFace; sourceInfo { name fixedWalls; } }
    { name obstacleFaces; type faceSet; action subset; source boxToFace; sourceInfo { box ($xmin $ymin -1) ($xmax $ymax 1); } }
);
EOT
cat > system/createPatchDict <<EOT
FoamFile { version 2.0; format ascii; class dictionary; object createPatchDict; }
pointSync false;
patches
(
    { name obstacle; patchInfo { type wall; } constructFrom set; set obstacleFaces; }
);
EOT
topoSet -latestTime > log.topoSet 2>&1
createPatch -overwrite > log.createPatch 2>&1

# createPatch (v1706) does not add the new patch to the fields: same conditions as the other walls,
# phi is left out (its fixedWalls size changed, simpleFoam computes it from U)
rm -f "$latest/phi"
$PYTHON - "$latest" <<'PY'
import sys
for field, condition in (("p", "zeroGradient"), ("U", "noSlip")):
    path = f"{sys.argv[1]}/{field}"
    text = open(path).read()
    entry = f"\n    obstacle\n    {{\n        type            {condition};\n    }}\n    fixedWalls\n"
    assert "\n    fixedWalls\n" in text, path
    open(path, "w").write(text.replace("\n    fixedWalls\n", entry, 1))
PY

cat >> system/controlDict <<EOT

functions
{
    // forces on the obstacle, rho = 1 (kinematic pressure)
    obstacleForces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (obstacle);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0 0 0);
    }

    wallShearStress
    {
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        patches         (obstacle);
        writeControl    timeStep;
        writeInterval   1;
    }

    // p and wall shear stress on the obstacle faces (face values, no interpolation)
    obstacleSurface
    {
        type            surfaces;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   1;
        surfaceFormat   raw;
        interpolationScheme cell;
        fields          (p wallShearStress);
        surfaces
        (
            obstacle { type patch; patches (obstacle); interpolate false; }
        );
    }

    // pressure drop
    inletPressure
    {
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        regionType      patch;
        name            inlet;
        operation       areaAverage;
        fields          (p);
    }

    outletPressure
    {
        \$inletPressure;
        name            outlet;
    }

    // cell fields for the peak vorticity, the dissipation and the recirculation zone
    vorticity
    {
        type            vorticity;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
    }

    gradU
    {
        type            grad;
        libs            ("libfieldFunctionObjects.so");
        field           U;
        writeControl    timeStep;
        writeInterval   1;
    }

    // U across the channel at the obstacle centroid (flow split) and 20 mm behind the obstacle (wake deficit).
    // A line stops where it enters the obstacle, so the centroid is sampled from both walls towards the obstacle.
    lines
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
            centroidBelow { type uniform; axis y; start ($x_centroid 0 0.005); end ($x_centroid 0.12 0.005); nPoints 2401; }
            centroidAbove { type uniform; axis y; start ($x_centroid 0.12 0.005); end ($x_centroid 0 0.005); nPoints 2401; }
            wakeLine      { type uniform; axis y; start ($x_wake 0 0.005); end ($x_wake 0.12 0.005); nPoints 2401; }
        );
    }
}
EOT
simpleFoam -postProcess -latestTime > log.postProcess 2>&1
postProcess -func writeCellCentres -latestTime > log.writeCellCentres 2>&1
postProcess -func writeCellVolumes -latestTime > log.writeCellVolumes 2>&1
echo "$latest"
