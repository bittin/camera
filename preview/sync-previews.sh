#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Copies freshly captured screenshots over the committed ones, but only where
# they actually differ.
#
# Two captures of an unchanged UI are not guaranteed to be byte-identical: a
# blinking recording dot, a timer reading 00:01 instead of 00:02, a pixel of
# antialiasing. Committing those would open a pull request on every push and
# train everyone to ignore preview PRs. A shot is only replaced when a
# meaningful share of its pixels changed.
#
# That share is measured two ways, and either one accepts the shot:
#
#   whole-frame  the share of differing pixels across the entire image, which
#                catches broad changes (a theme, a relaid-out panel)
#   worst tile   the same share computed over a 16x16 grid, keeping the worst
#                cell, which catches a small element that is entirely wrong
#
# The tile pass exists because the whole-frame share alone is a function of how
# *large* the changed element is, not how wrong it is. A corrupt gallery
# thumbnail covers about 0.25% of a 900x700 shot, under any whole-frame
# threshold loose enough to absorb a blinking dot, so a screenshot showing a
# green-smeared thumbnail was silently kept over a correct fresh capture. Within
# its own tile that same thumbnail is a ~40% change, far above the timer digit
# or antialiasing it has to be told apart from.
#
# Usage:
#   preview/sync-previews.sh <new-dir> <committed-dir>
#
# Environment:
#   PREVIEW_FUZZ        per-pixel colour tolerance, ImageMagick syntax (default 2%)
#   PREVIEW_THRESHOLD   share of differing pixels needed to accept a shot (default 0.005 = 0.5%)
#   PREVIEW_TILE_THRESHOLD
#                       share of differing pixels within the worst grid cell
#                       needed to accept a shot (default 0.25 = 25%)
#   PREVIEW_TILES       grid resolution for the tile pass (default 16)
#   PREVIEW_OPTIMIZE    set to 0 to commit the PNGs exactly as captured
#   PREVIEW_OXIPNG_LEVEL
#                       oxipng optimization level (default 4)

set -euo pipefail

NEW_DIR="${1:?usage: sync-previews.sh <new-dir> <committed-dir>}"
OLD_DIR="${2:?usage: sync-previews.sh <new-dir> <committed-dir>}"

FUZZ="${PREVIEW_FUZZ:-2%}"
THRESHOLD="${PREVIEW_THRESHOLD:-0.005}"
TILE_THRESHOLD="${PREVIEW_TILE_THRESHOLD:-0.25}"
TILES="${PREVIEW_TILES:-16}"
OXIPNG_LEVEL="${PREVIEW_OXIPNG_LEVEL:-4}"

command -v magick >/dev/null || command -v convert >/dev/null || {
    echo "error: ImageMagick is required" >&2
    exit 1
}

# ImageMagick 7 folded the tools into one `magick` command; support both.
if command -v magick >/dev/null; then
    im_convert() { magick "$@"; }
    im_identify() { magick identify "$@"; }
else
    im_convert() { convert "$@"; }
    im_identify() { identify "$@"; }
fi

# Prints the share of pixels (0..1) that differ by more than FUZZ.
#
# Deliberately not `compare -metric AE`: on an HDRI build that returns a
# scaled error sum in scientific notation rather than a pixel count, which a
# previous version of this script parsed as a small integer and reported every
# screenshot as unchanged, including ones that were entirely wrong.
#
# Instead: absolute per-channel difference, flattened to grey, thresholded at
# FUZZ. Every pixel is then black (within tolerance) or white (differs), so the
# mean of the result *is* the share of differing pixels.
diff_share() {
    im_convert "$1" "$2" -compose Difference -composite \
        -colorspace Gray -threshold "$FUZZ" -format '%[fx:mean]' info:
}

# Prints the share of differing pixels (0..1) within the worst cell of a
# TILES x TILES grid.
#
# `-scale` to a fixed grid averages each cell, so on the same black/white
# thresholded diff every output pixel is that cell's share of differing pixels
# and the maximum is the worst cell. `!` forces the exact grid regardless of
# aspect ratio.
diff_tile_share() {
    im_convert "$1" "$2" -compose Difference -composite \
        -colorspace Gray -threshold "$FUZZ" -scale "${TILES}x${TILES}!" \
        -format '%[fx:maxima]' info:
}

# A comparison that fails must never look like "nothing changed": that is how
# a broken harness quietly keeps stale screenshots. The value may be in
# scientific notation (1.5873e-06 is a single differing pixel), so accept that
# spelling; awk parses it correctly.
require_share() {
    if ! [[ "$1" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "error: could not compare $2 (got '$1')" >&2
        exit 1
    fi
}

as_pct() { awk "BEGIN { printf \"%.2f\", $1 * 100 }"; }

# Losslessly recompresses the freshly captured shots, in place, before anything
# else looks at them.
#
# The repository carries over a hundred of these PNGs and CI regenerates them on
# every push to main that touches src/**, so a byte saved here is saved again in
# every future regeneration: git keeps every generation of every blob forever.
# oxipng re-derives the per-row filters and re-deflates the image data without
# touching a single pixel, which is worth about 10% on these shots (measured
# over a 14 shot sample spanning the top level, variants/ and locales/) for well
# under a minute of wall clock across the whole set.
#
# Why this runs BEFORE the pixel comparison rather than after the copy:
#
#   * once a shot has been replaced, the committed file is itself oxipng output,
#     so optimizing the fresh capture first keeps both sides of the comparison
#     in the same representation instead of comparing "as captured" against
#     "as committed"
#   * whatever was compared is exactly what gets committed, so a shot can never
#     be judged on one byte stream and stored as another
#
# Neither would be safe with a lossy optimizer, and it is worth spelling out why
# this step is lossless. `oxipng -o4 --strip safe` was verified over that same
# sample to decode to pixel-identical output (`magick compare -metric AE` = 0),
# so it cannot move either share below by even one pixel.
#
# Quantization (pngquant --quality=70-95) was measured as well and deliberately
# rejected despite saving far more, about 66%. It is wrong here twice over:
#
#   * it dithers the viewfinder photo. The smooth sky gradient picks up visible
#     mottling and faint banding, and these shots exist to sell the app on
#     Flathub.
#   * it defeats the change detection this whole script is built around. The
#     palette is chosen from the whole frame, so a change to 0.04% of the pixels
#     (one timer digit) re-picked it and moved 2.9% of *all* pixels, past the
#     0.5% whole-frame threshold, with a worst tile of 33% past the 25% tile
#     threshold. Every shot would be rewritten on every run, which is precisely
#     the churn the thresholds exist to prevent.
#
# oxipng is installed in the container image (see preview/Containerfile) but is
# treated as optional: someone running this script on a host without it should
# still get a correct sync, only larger files.
optimize_shots() {
    (( $# )) || return 0

    if [[ "${PREVIEW_OPTIMIZE:-1}" != "1" ]]; then
        echo "note: PREVIEW_OPTIMIZE=0, committing PNGs as captured"
        return 0
    fi

    if ! command -v oxipng >/dev/null; then
        echo "note: oxipng not found, committing PNGs as captured (about 10% larger)" >&2
        return 0
    fi

    # One invocation for the whole set rather than one per shot: oxipng
    # parallelises across the files it is given, which measured about 2.5x
    # faster than a call per file.
    #
    # A failure must not fail the sync. The optimizer is a size tweak, not a
    # correctness step, and a partially optimized set is still a set of valid,
    # pixel-identical PNGs.
    echo "==> Optimizing $# captured shots with oxipng -o $OXIPNG_LEVEL"
    if ! oxipng -o "$OXIPNG_LEVEL" --strip safe --quiet "$@"; then
        echo "warning: oxipng reported an error, continuing with the shots as they are" >&2
    fi
}

changed=0
unchanged=0
added=0

shopt -s nullglob
# The published English shots sit at the top level, the appearance variants under
# variants/, and one published shot per translated language under
# locales/<lang>/. All are synced the same way, so neither the gallery in
# preview/README.md nor the localized screenshots in metainfo.xml drift from the
# shot they illustrate.
#
# Collected into an array first so the optimizer can be handed the whole set in
# one go, and so it cannot possibly run over a different set of files than the
# comparison loop below.
shots=( "$NEW_DIR"/preview-*.png "$NEW_DIR"/variants/preview-*.png \
        "$NEW_DIR"/locales/*/preview-*.png )

optimize_shots "${shots[@]}"

for new in "${shots[@]}"; do
    name="${new#"$NEW_DIR"/}"
    old="$OLD_DIR/$name"
    mkdir -p "$(dirname "$old")"

    if [[ ! -f "$old" ]]; then
        cp "$new" "$old"
        echo "added    $name (no committed version)"
        added=$(( added + 1 ))
        continue
    fi

    new_size="$(im_identify -format '%wx%h' "$new")"
    old_size="$(im_identify -format '%wx%h' "$old")"
    if [[ "$new_size" != "$old_size" ]]; then
        cp "$new" "$old"
        echo "changed  $name (size ${old_size} -> ${new_size})"
        changed=$(( changed + 1 ))
        continue
    fi

    share="$(diff_share "$old" "$new")"
    require_share "$share" "$name"

    tile_share="$(diff_tile_share "$old" "$new")"
    require_share "$tile_share" "$name"

    if awk "BEGIN { exit !($share > $THRESHOLD) }"; then
        cp "$new" "$old"
        printf 'changed  %s (%s%% of pixels)\n' "$name" "$(as_pct "$share")"
        changed=$(( changed + 1 ))
    elif awk "BEGIN { exit !($tile_share > $TILE_THRESHOLD) }"; then
        cp "$new" "$old"
        printf 'changed  %s (%s%% of pixels overall, but %s%% within one tile)\n' \
            "$name" "$(as_pct "$share")" "$(as_pct "$tile_share")"
        changed=$(( changed + 1 ))
    else
        printf 'unchanged %s (%s%% of pixels, %s%% worst tile, below thresholds)\n' \
            "$name" "$(as_pct "$share")" "$(as_pct "$tile_share")"
        unchanged=$(( unchanged + 1 ))
    fi
done

echo
printf '%s changed, %s added, %s unchanged (fuzz %s, threshold %s%%, tile threshold %s%% on a %sx%s grid)\n' \
    "$changed" "$added" "$unchanged" "$FUZZ" "$(as_pct "$THRESHOLD")" "$(as_pct "$TILE_THRESHOLD")" "$TILES" "$TILES"
