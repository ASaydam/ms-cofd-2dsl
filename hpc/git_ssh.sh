#!/bin/bash
# ssh for git (GIT_SSH, git 1.8 has no GIT_SSH_COMMAND) when hpc/publish_status.py pushes the status: only the deploy
# key of the status repository and the checked host keys of github.com
exec ssh -i "$MSCOFD_HOME/config/github_status_key" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$MSCOFD_HOME/config/github_known_hosts" "$@"
