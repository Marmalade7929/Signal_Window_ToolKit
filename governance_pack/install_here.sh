#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install this governance pack into the current repository.

Usage:
  bash governance_pack/install_here.sh [--dry-run] [--force] [target-path]

Defaults:
  target-path defaults to the current directory.
EOF
}

DRY_RUN=0
FORCE=0
TARGET_PATH="."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      TARGET_PATH="$1"
      shift
      ;;
  esac
done

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$(cd "$TARGET_PATH" && pwd)"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Target path does not exist: $TARGET_DIR" >&2
  exit 1
fi

COPIED=()
SKIPPED=()

copy_file() {
  local src="$1"
  local rel="$2"
  local dst="$TARGET_DIR/$rel"

  mkdir -p "$(dirname "$dst")"

  if [[ -e "$dst" && $FORCE -eq 0 ]]; then
    SKIPPED+=("$rel")
    return
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    COPIED+=("$rel (dry-run)")
    return
  fi

  cp "$src" "$dst"
  COPIED+=("$rel")
}

while IFS= read -r -d '' src; do
  rel="${src#$PACK_DIR/}"
  copy_file "$src" "$rel"
done < <(find "$PACK_DIR/governance" -type f -print0)

while IFS= read -r -d '' src; do
  rel="${src#$PACK_DIR/}"
  copy_file "$src" "$rel"
done < <(find "$PACK_DIR/skills" -type f -print0)

echo "Pack dir:   $PACK_DIR"
echo "Target dir: $TARGET_DIR"

echo
echo "Copied/updated (${#COPIED[@]}):"
if [[ ${#COPIED[@]} -eq 0 ]]; then
  echo "  - none"
else
  for item in "${COPIED[@]}"; do
    echo "  - $item"
  done
fi

echo
echo "Skipped (${#SKIPPED[@]}):"
if [[ ${#SKIPPED[@]} -eq 0 ]]; then
  echo "  - none"
else
  for item in "${SKIPPED[@]}"; do
    echo "  - $item"
  done
fi

echo
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete. No files were written."
elif [[ ${#COPIED[@]} -eq 0 ]]; then
  echo "No changes were made."
else
  echo "Governance pack installed."
fi
