#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Runner image ubuntu24/20260726.254 paired Podman 5.8.4 with a crun
# that failed with "unknown version specified". Probe rather than assuming
# that the runner still needs the runc fallback.
set -euo pipefail
command -v podman >/dev/null 2>&1 || { sudo apt-get update && sudo apt-get install -y podman; }
echo "podman: $(podman --version)"
echo "crun:   $(crun --version 2>/dev/null | head -1 || echo 'not installed')"
echo "runc:   $(runc --version 2>/dev/null | head -1 || echo 'not installed')"

# A pull failure (Docker Hub rate limits) must not be mistaken for a
# broken runtime, so only the `run` result decides anything.
probe=docker.io/library/busybox:latest
if ! podman pull -q "$probe" >/dev/null 2>&1; then
  echo "::warning::could not pull $probe; skipping OCI runtime check"
  exit 0
fi

if podman run --rm "$probe" true; then
  echo "Default OCI runtime works"
  exit 0
fi

echo "Default OCI runtime cannot start a container; falling back to runc"
if ! command -v runc >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y runc
fi
mkdir -p ~/.config/containers
printf '[engine]\nruntime = "runc"\n' > ~/.config/containers/containers.conf
# Fail here rather than 10 minutes into the build if runc is no better.
podman run --rm "$probe" true
echo "Using runc"
