#!/bin/bash
#
# Copies a committed revision of this repository to the cluster (run on your own computer, in the repository):
#
#   bash hpc/deploy.sh                 # HEAD
#   bash hpc/deploy.sh <revision> <ssh host>
#
# Only committed code runs in the production: the files come from git archive, and REVISION holds the commit, which
# every task records in its provenance. The cluster needs no git credentials. The new copy replaces
# hpc_run/ms-cofd-2dsl/repo in one rename; running jobs keep the files they have open.

set -euo pipefail
revision=${1:-HEAD}
host=${2:-kuacc-claude}

commit=$(git rev-parse "$revision")
if [ "$revision" = "HEAD" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "uncommitted changes are not deployed, only $commit"
fi

git archive --format=tar "$commit" | ssh "$host" "set -e
    target=/kuacc/users/\$USER/hpc_run/ms-cofd-2dsl/repo
    rm -rf \"\$target.new\" && mkdir -p \"\$target.new\"
    tar -xf - -C \"\$target.new\"
    echo $commit > \"\$target.new/REVISION\"
    rm -rf \"\$target.old\"
    if [ -d \"\$target\" ]; then mv \"\$target\" \"\$target.old\"; fi
    mv \"\$target.new\" \"\$target\"
    rm -rf \"\$target.old\"
    echo \"deployed $commit to \$target\""
