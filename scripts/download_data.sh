#!/usr/bin/env bash
set -euo pipefail
# Download large SHARP benchmark AnnData files from the GitHub Release.
REPO="${REPO:-Meteor109/SHARP}"
TAG="${TAG:-datasets-v1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/E18.5_mouse_brain" "$ROOT/data/Stereo"
TMPDIR="$(mktemp -d)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" \
    --pattern 'E18_adata_rna.h5ad' \
    --pattern 'Stereo_AB.h5ad' \
    --pattern 'Stereo_DPI.h5ad' \
    --dir "$TMPDIR" --clobber
else
  BASE="https://github.com/${REPO}/releases/download/${TAG}"
  curl -fL "$BASE/E18_adata_rna.h5ad" -o "$TMPDIR/E18_adata_rna.h5ad"
  curl -fL "$BASE/Stereo_AB.h5ad" -o "$TMPDIR/Stereo_AB.h5ad"
  curl -fL "$BASE/Stereo_DPI.h5ad" -o "$TMPDIR/Stereo_DPI.h5ad"
fi

mv -f "$TMPDIR/E18_adata_rna.h5ad" "$ROOT/data/E18.5_mouse_brain/"
mv -f "$TMPDIR/Stereo_AB.h5ad" "$ROOT/data/Stereo/"
mv -f "$TMPDIR/Stereo_DPI.h5ad" "$ROOT/data/Stereo/"
echo "Downloaded AnnData files into ./data"
