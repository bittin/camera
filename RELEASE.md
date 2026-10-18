# Release Guide

This document describes how to create a new release of Camera.

## Overview

The release process uses two GitHub Actions workflows:

1. **Create Release** (`create-release.yml`) - Manually triggered to prepare and tag a release
2. **Release** (`release.yml`) - Automatically triggered when a tag is pushed, builds and publishes the release

## Creating a New Release

### Step 1: Go to GitHub Actions

Navigate to the repository's **Actions** tab and select the **"Create Release"** workflow from the left sidebar.

### Step 2: Run the Workflow

Click **"Run workflow"** and you'll see an input field:

| Field | Description |
|-------|-------------|
| **Version number** | Optional. Leave empty to auto-increment the patch version (e.g., `0.1.6` → `0.1.7`). Or specify a version like `0.2.0` or `1.0.0`. |

Click **"Run workflow"** to start.

### Step 3: What Happens Automatically

The workflow will:

1. **Determine the version** - Uses your input or auto-increments from the latest tag
2. **Generate release notes** - Fetches "What's Changed" from merged PRs since the last release
3. **Update release metadata** - Updates the package version in Cargo.toml, adds a `<release>` entry to metainfo.xml, updates Cargo.lock, and regenerates cargo-sources.json
4. **Commit the changes** - Pushes those four files to the selected workflow branch (normally main)
5. **Create the git tag** - Tags the commit with the new version (e.g., `v0.1.7`)

This triggers the **Release** workflow, which:

1. **Builds binaries** - For x86_64, aarch64, armhf, and riscv64 with glibc, plus x86_64 and aarch64 with musl
2. **Builds Flatpak bundles** - For x86_64; aarch64 is built only on a manual Release run with `build_aarch64_flatpak` enabled
3. **Generates APKBUILD** - Alpine Linux package recipe with correct sha512sum
4. **Creates GitHub Release** - With all artifacts and release notes

Flathub publication is maintained separately; these workflows do not update the Flathub repository.

## Version Numbering

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (x.0.0) - Incompatible API/behavior changes
- **MINOR** (0.x.0) - New features, backward compatible
- **PATCH** (0.0.x) - Bug fixes, backward compatible

Examples:
- `0.1.7` - Patch release with bug fixes
- `0.2.0` - Minor release with new features
- `1.0.0` - Major stable release

## Prerequisites

### For GitHub Releases

Configure the repository secret `RELEASE_TOKEN` with permission to push release
commits and tags. **Create Release** uses it at checkout: a tag pushed with the
default `GITHUB_TOKEN` would not trigger the **Release** workflow. The **Release**
workflow itself uses `GITHUB_TOKEN` with `contents: write` to publish artifacts.

### For Flathub Publishing

Update the [Flathub repository](https://github.com/flathub/io.github.cosmic_utils.camera)
separately. There is no `FLATHUB_TOKEN` integration in these workflows. Keep the
committed `cargo-sources.json` synchronized with Cargo.lock before tagging.

## Troubleshooting

### "Tag already exists" Error
The workflow will fail if you try to create a version that already exists. Check existing tags:
```bash
git tag --sort=-v:refname | head -10
```

### Release Notes Are Empty
Release notes are generated from merged pull requests using merge dates rather than merge-commit SHAs. During **Create Release** preparation, no matching PRs produces the fallback "Bug fixes and improvements." The final **Release** workflow reconstructs the PR list independently and appends installation instructions; it does not use that fallback.

### Flathub Build Fails
If a separately submitted Flathub update fails:
1. Check the [Flathub Buildbot](https://buildbot.flathub.org/#/apps/io.github.cosmic_utils.camera)
2. Common issues: missing dependencies, cargo source hash mismatches
3. You may need to manually fix the Flathub repository

## Manual Release (Emergency)

If the automated workflow fails, you can create a release manually:

```bash
# 1. Update the package version in Cargo.toml and the version/date/changelog
#    in resources/io.github.cosmic_utils.camera.metainfo.xml.
# 2. Refresh the package lock entry and generated Flatpak sources.
cargo update --workspace
just flatpak-cargo-sources
just check
# 3. Commit the release inputs together.
git add Cargo.toml Cargo.lock cargo-sources.json resources/io.github.cosmic_utils.camera.metainfo.xml
git commit -m "Release v0.1.7: update version and metainfo"
git push origin main

# 4. Create and push the tag
git tag -a v0.1.7 -m "Release v0.1.7"
git push origin v0.1.7
```

This will trigger the Release workflow to build and publish.

## Release Artifacts

Each release includes:

| Artifact | Description |
|----------|-------------|
| `camera-x86_64-linux.tar.gz` | Linux binary for x86_64 |
| `camera-aarch64-linux.tar.gz` | Linux binary for ARM64 |
| `camera-armhf-linux.tar.gz` | Linux binary for armhf (32-bit) |
| `camera-riscv64-linux.tar.gz` | Linux binary for RISC-V 64 |
| `camera-x86_64-musl-linux.tar.gz` | Linux binary for x86_64 (musl, dynamically linked) |
| `camera-aarch64-musl-linux.tar.gz` | Linux binary for ARM64 (musl, dynamically linked) |
| `camera-x86_64.flatpak` | Flatpak bundle for x86_64 |
| `camera-aarch64.flatpak` | Flatpak bundle for ARM64 (manual opt-in only) |
| `camera-vX.Y.Z-source.zip` | Source code archive |
| `APKBUILD` | Alpine Linux package build recipe |
