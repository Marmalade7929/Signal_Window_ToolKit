#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install governance starter pack into another repository.

Usage:
  scripts/install_governance_pack.sh [--dry-run] [--force] [--with-curation-skill] <target-repo-path>

Options:
  --dry-run              Show planned actions without writing files.
  --force                Overwrite existing files in the target repository.
  --with-curation-skill  Also copy skills/curate-public-repo-output.
  -h, --help             Show this help message.
EOF
}

FORCE=0
DRY_RUN=0
WITH_CURATION=0
TARGET_REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --with-curation-skill)
      WITH_CURATION=1
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
      if [[ -n "$TARGET_REPO" ]]; then
        echo "Only one target path is allowed." >&2
        usage
        exit 1
      fi
      TARGET_REPO="$1"
      shift
      ;;
  esac
done

if [[ -z "$TARGET_REPO" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$TARGET_REPO" ]]; then
  echo "Target directory does not exist: $TARGET_REPO" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_REPO="$(cd "$TARGET_REPO" && pwd)"

GOVERNANCE_FILES=(
  "governance/publication_policy.md"
  "governance/publish_checklist.md"
  "governance/public_commit_pr_strategy.md"
  "governance/public_migration_backlog.md"
  "governance/deployment_strategy.md"
  "governance/deployment_strategy_template.md"
)

SKILL_ROOTS=(
  "skills/repo-governance-assistant"
)

if [[ $WITH_CURATION -eq 1 ]]; then
  SKILL_ROOTS+=("skills/curate-public-repo-output")
fi

COPIED=()
SKIPPED=()
MISSING=()

copy_file() {
  local rel_path="$1"
  local src_path="$SOURCE_REPO_ROOT/$rel_path"
  local dst_path="$TARGET_REPO/$rel_path"

  if [[ ! -f "$src_path" ]]; then
    MISSING+=("$rel_path")
    return
  fi

  mkdir -p "$(dirname "$dst_path")"

  if [[ -e "$dst_path" && $FORCE -eq 0 ]]; then
    SKIPPED+=("$rel_path")
    return
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    COPIED+=("$rel_path (dry-run)")
    return
  fi

  cp "$src_path" "$dst_path"
  COPIED+=("$rel_path")
}

ensure_file() {
  local rel_path="$1"
  local dst_path="$TARGET_REPO/$rel_path"

  mkdir -p "$(dirname "$dst_path")"

  if [[ -e "$dst_path" && $FORCE -eq 0 ]]; then
    SKIPPED+=("$rel_path")
    return
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    COPIED+=("$rel_path (dry-run)")
    return
  fi

  : > "$dst_path"
  COPIED+=("$rel_path")
}

for file in "${GOVERNANCE_FILES[@]}"; do
  copy_file "$file"
done

for skill_root in "${SKILL_ROOTS[@]}"; do
  source_skill_root="$SOURCE_REPO_ROOT/$skill_root"

  if [[ ! -d "$source_skill_root" ]]; then
    MISSING+=("$skill_root/")
    continue
  fi

  while IFS= read -r -d '' skill_file; do
    rel_file="${skill_file#$SOURCE_REPO_ROOT/}"
    copy_file "$rel_file"
  done < <(find "$source_skill_root" -type f -print0)
done

ensure_file "governance/signoffs/.gitkeep"

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "Missing source files/directories:" >&2
  for item in "${MISSING[@]}"; do
    echo "  - $item" >&2
  done
  exit 1
fi

echo "Source repo: $SOURCE_REPO_ROOT"
echo "Target repo: $TARGET_REPO"

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
  echo "Governance starter pack installed."
fi
