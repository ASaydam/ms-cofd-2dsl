---
license: cc-by-4.0
pretty_name: MS-COFD-2DSL archived OpenFOAM cases
size_categories:
- 10K<n<100K
tags:
- cfd
- openfoam
- fluid-dynamics
- multi-fidelity
- laminar-flow
- simulation
configs:
- config_name: index
  data_files: index.csv
---

# MS-COFD-2DSL: archived OpenFOAM cases

MS-COFD-2DSL (Multi-Scale Central Object Flow Dataset, 2D Steady Laminar) holds 4,500 two-dimensional, steady,
laminar channel flows past one obstacle. Each flow was solved with OpenFOAM in five configurations (runs). This
repository holds the complete OpenFOAM case of every run, one archive per run. The arrays and tables made from them
(fields on regular grids, metadata, attributes, splits) are a separate dataset:
[asaydam/ms-cofd-2dsl](https://huggingface.co/datasets/asaydam/ms-cofd-2dsl).

## Contents

```
<run>/<case>.tar.zst    one archive per run and case: 5 runs × 4,500 cases = 22,500 archives, about 1.56 TB
index.csv               run, case, file, bytes and md5 of every archive
```

| Run | Mesh | Convection |
|---|---|---|
| `0.25x` | 2 mm | second order |
| `base_fo` | 1 mm | first order |
| `base_lu` | 1 mm | second order |
| `4x` | 0.5 mm | second order |
| `16x` | 0.25 mm | second order |

The case names (for example `normal_rectangle_0001`) are those of the processed dataset. Its metadata records the
settings, starting fields, iterations and residuals of every run. Each archive holds two folders:

- `<case>/`, the case of the run: `system/`, `constant/` with the mesh, the gmsh input and mesh (`mesh.geo`,
  `mesh.msh`), the initial fields `0/`, the last two written time steps, the logs of every step (`log.*`) and the run
  record `provenance.json`.
- `<case>_attributes/`, the copy of the case in which the attributes were computed, with the obstacle as its own
  patch, OpenFOAM's function-object output in `postProcessing/` (forces, obstacle surface, inlet and outlet pressure,
  line samples) and its logs.

The logs and run records keep the paths, host names and SLURM job IDs of the cluster that ran the simulations. They
contain no passwords, keys or tokens.

## Download and open

```python
from huggingface_hub import snapshot_download
snapshot_download("asaydam/ms-cofd-2dsl-raw", repo_type="dataset", local_dir="ms-cofd-2dsl-raw",
                  allow_patterns=["index.csv", "16x/normal_rectangle_0001.tar.zst"])    # or "base_lu/*" for a run
```

```bash
cd ms-cofd-2dsl-raw
awk -F, 'NR > 1 {print $5 "  " $1 "/" $3}' index.csv | md5sum -c --ignore-missing --quiet    # checks the MD5s
zstd -dc 16x/normal_rectangle_0001.tar.zst | tar -xf -        # or tar --zstd -xf (GNU tar 1.31 or later)
```

## Working with a case

The archives are complete OpenFOAM cases (OpenFOAM v2512, ESI). They can be opened in the container image of the
[pipeline](https://github.com/ASaydam/ms-cofd-2dsl), `ghcr.io/asaydam/ms-cofd-2dsl`, with the software versions that
made them, to compute further quantities or other resamplings. The image digest of a run is `container_digest` in
its `provenance.json`, and `/opt/mscofd/activate.sh` loads OpenFOAM and the Python environment:

```bash
docker run --rm -v "$PWD:/data" --entrypoint bash ghcr.io/asaydam/ms-cofd-2dsl@<container_digest> \
    -c "source /opt/mscofd/activate.sh && foamListTimes -case /data/normal_rectangle_0001"
```

## Citation and licence

If you use these archives, please cite the dataset paper: [citation of the dataset paper].

The archives are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): you may share and adapt
them for any purpose, including commercially, if you give credit by citing the paper and indicate any changes.
