#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Regenerates preview/preview-0NN.png inside the pinned container image.
#
# This is the entry point for both humans and CI: the same image, the same
# compositor and the same fonts produce the same pixels, so a diff in the
# committed screenshots means the UI actually changed.
#
# Usage:
#   preview/generate-previews.sh            # regenerate all previews
#   SHOTS=001,004 preview/generate-previews.sh
#
# Environment:
#   CONTAINER_ENGINE  podman (default) or docker
#   PREVIEW_IMAGE     image tag to build and run (default camera-previews)
#   NO_BUILD_IMAGE    set to 1 to reuse an already-built image
#   SHOTS             comma-separated shot numbers, passed through
#   PREVIEW_FUZZ / PREVIEW_THRESHOLD / PREVIEW_TILE_THRESHOLD / PREVIEW_TILES
#   PREVIEW_OPTIMIZE / PREVIEW_OXIPNG_LEVEL
#                     forwarded to sync-previews.sh, which documents and
#                     defaults them. Set PREVIEW_THRESHOLD=0
#                     PREVIEW_TILE_THRESHOLD=0 to take every shot that differs
#                     by at least one pixel, which is what the "force" input on
#                     the Regenerate Preview Screenshots workflow does.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENGINE="${CONTAINER_ENGINE:-}"
if [[ -z "$ENGINE" ]]; then
    if command -v podman >/dev/null; then
        ENGINE=podman
    elif command -v docker >/dev/null; then
        ENGINE=docker
    else
        echo "error: neither podman nor docker is installed" >&2
        exit 1
    fi
fi

IMAGE="${PREVIEW_IMAGE:-camera-previews}"

# The app fakes its own camera in CPU memory (--preview-fake-camera plus
# --preview-source), so no dma-buf provider (/dev/udmabuf or a dma_heap) and no
# device access is needed. This is what lets the harness run on GitHub-hosted
# runners, whose kernel ships no udmabuf module.

# Cargo's registry and target directory live outside the repo tree so a
# container build never fights with the host toolchain over target/.
CACHE_DIR="$REPO_DIR/.preview-cache"
mkdir -p "$CACHE_DIR/cargo" "$CACHE_DIR/target"

if [[ "${NO_BUILD_IMAGE:-0}" != "1" ]]; then
    echo "==> Building $IMAGE with $ENGINE"
    "$ENGINE" build -t "$IMAGE" -f "$SCRIPT_DIR/Containerfile" "$REPO_DIR"
fi

run_args=(
    --rm
    -v "$REPO_DIR:/src:Z"
    -v "$CACHE_DIR/cargo:/cargo:Z"
    -v "$CACHE_DIR/target:/target:Z"
    -e CARGO_HOME=/cargo
    -e CARGO_TARGET_DIR=/target
    -e "SHOTS=${SHOTS:-}"
    -e "KEEP_GOING=${KEEP_GOING:-0}"
)

# The comparison knobs, forwarded only when the caller actually set one, so
# sync-previews.sh stays the single source of truth for their defaults. It runs
# inside the container, so a knob that is not forwarded here is silently ignored
# no matter where it was set — which is how `PREVIEW_TILE_THRESHOLD` came to be
# documented and unreachable, leaving the tile pass pinned at its default while
# the whole-frame threshold tracked the environment.
for var in PREVIEW_FUZZ PREVIEW_THRESHOLD PREVIEW_TILE_THRESHOLD PREVIEW_TILES \
           PREVIEW_OPTIMIZE PREVIEW_OXIPNG_LEVEL; do
    if [[ -n "${!var:-}" ]]; then
        run_args+=( -e "$var=${!var}" )
    fi
done

# Rootless podman maps the container's root to the invoking user, so files land
# with the right ownership. Docker does not, so run as the caller there.
if [[ "$ENGINE" == "docker" ]]; then
    run_args+=( --user "$(id -u):$(id -g)" -e HOME=/tmp/home )
fi

echo "==> Capturing previews"
"$ENGINE" run "${run_args[@]}" "$IMAGE" \
    -c 'preview/capture-previews.sh /tmp/shots && preview/sync-previews.sh /tmp/shots preview'

echo
echo "==> Done. Review with: git diff --stat preview/"
