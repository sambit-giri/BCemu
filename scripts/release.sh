#!/usr/bin/env bash
# Build, sanity-check, tag, upload to PyPI, and create a matching GitHub
# release — all in one step. Run manually whenever you want to cut a
# release; ordinary commits do not trigger this.
#
# Usage:
#   scripts/release.sh                build + check + tag + upload + release
#   scripts/release.sh --dry-run      build + check only, nothing published
#   scripts/release.sh --yes          skip the y/n prompts before each
#                                      irreversible step (tag push, PyPI
#                                      upload, GitHub release)
#   scripts/release.sh --notes "..."  use these GitHub release notes instead
#                                      of --generate-notes
#
# Requires: python3 -m pip install build twine; gh CLI authenticated;
# ~/.pypirc (or TWINE_* env vars) configured for PyPI.

set -euo pipefail

DRY_RUN=0
ASSUME_YES=0
NOTES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --notes) NOTES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

confirm() {
  if [[ $ASSUME_YES -eq 1 ]]; then return 0; fi
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

VERSION=$(python3 -c "import re; print(re.search(r\"version='([^']+)'\", open('setup.py').read()).group(1))")
TAG="v${VERSION}"
echo "Package version (setup.py): ${VERSION}  (tag ${TAG})"

# --- sanity checks ----------------------------------------------------

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean — commit or stash changes first:" >&2
  git status --short
  exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Git tag ${TAG} already exists locally. Bump the version in setup.py first." >&2
  exit 1
fi

if git ls-remote --tags origin 2>/dev/null | grep -q "refs/tags/${TAG}$"; then
  echo "Tag ${TAG} already exists on origin. Bump the version in setup.py first." >&2
  exit 1
fi

if gh release view "$TAG" >/dev/null 2>&1; then
  echo "GitHub release ${TAG} already exists. Bump the version in setup.py first." >&2
  exit 1
fi

if python3 -m pip index versions BCemu 2>/dev/null | head -1 | grep -qF "(${VERSION})"; then
  echo "Warning: PyPI already appears to list version ${VERSION} as latest." >&2
  confirm "Continue anyway?" || exit 1
fi

# --- build --------------------------------------------------------------

echo "Cleaning old build artifacts..."
rm -rf dist build src/*.egg-info

echo "Building sdist + wheel..."
python3 -m build

echo "Running twine check..."
python3 -m twine check dist/*

# Guard against the input_data greedy-glob regression (see git history):
# the package should stay small — large model weights are fetched on
# demand via download.py, not bundled.
WHEEL="$(ls dist/*-py3-none-any.whl)"
WHEEL_SIZE=$(stat -f%z "$WHEEL" 2>/dev/null || stat -c%s "$WHEEL")
MAX_SIZE=$((5 * 1024 * 1024))  # 5 MB
if [[ "$WHEEL_SIZE" -gt "$MAX_SIZE" ]]; then
  echo "Wheel is ${WHEEL_SIZE} bytes (> 5MB) — this usually means" >&2
  echo "input_data/ swept in files meant to be downloaded on demand." >&2
  echo "Check setup.py's package_data before releasing." >&2
  exit 1
fi
echo "Wheel size OK: ${WHEEL_SIZE} bytes ($WHEEL)."

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete. Artifacts in dist/, nothing tagged/uploaded/released."
  exit 0
fi

# --- tag & push -----------------------------------------------------------

confirm "Tag ${TAG} and push to origin?" || exit 1
git tag -a "$TAG" -m "Release ${TAG}"
git push origin "$TAG"

# --- PyPI upload ------------------------------------------------------

confirm "Upload dist/* to PyPI as ${VERSION}?" || exit 1
python3 -m twine upload dist/*

# --- GitHub release -----------------------------------------------------

confirm "Create GitHub release ${TAG}?" || exit 1
if [[ -z "$NOTES" ]]; then
  gh release create "$TAG" dist/*.whl dist/*.tar.gz --title "$TAG" --generate-notes
else
  gh release create "$TAG" dist/*.whl dist/*.tar.gz --title "$TAG" --notes "$NOTES"
fi

echo
echo "Released ${TAG}:"
echo "  https://pypi.org/project/BCemu/${VERSION}/"
echo "  $(gh release view "$TAG" --json url -q .url)"
