# Software environment of the MS-COFD-2DSL pipeline: OpenFOAM (ESI) and a Python environment (micromamba,
# conda-forge) with gmsh, VTK and the pipeline's Python packages.
#
# Built by .github/workflows/container.yml and pushed to ghcr.io/asaydam/ms-cofd-2dsl; on the cluster it is pulled
# with Apptainer (see hpc/env.sh). The exact versions of every build are written to /opt/mscofd/versions.

# OpenFOAM v2512 (ESI / opencfd), pinned by digest
FROM opencfd/openfoam-default:2512@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# system tools: time (peak memory of the solver), zstd (case archives)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl bzip2 time zstd \
    && rm -rf /var/lib/apt/lists/*

# micromamba and the Python environment (conda-forge only)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba
COPY environment/environment.yml /opt/mscofd/environment.yml
RUN micromamba create -y -p /opt/conda/envs/mscofd -f /opt/mscofd/environment.yml \
    && micromamba clean -a -y

# activation script: OpenFOAM and the Python environment, sourced by every pipeline script in the container
COPY environment/activate.sh /opt/mscofd/activate.sh
COPY environment/record_versions.sh /opt/mscofd/record_versions.sh
RUN bash /opt/mscofd/record_versions.sh /opt/mscofd/versions

ENV MSCOFD_CONTAINER=1
