#!/usr/bin/env bash
set -u

OUT="PROJECT_STATE_SUMMARY.md"
ROOT="$(pwd)"

# Limits to avoid absurdly huge output
MAX_FILE_LINES=5
MAX_TEXT_PREVIEW=10
MAX_FILES_LIST=300

# Extensions treated as code/text for keyword inspection
TEXT_EXT_REGEX='\.(py|ipynb|md|txt|yaml|yml|json|toml|ini|cfg|csv|tsv|sh|bash|zsh|log|tex)$'

# Directories to ignore
PRUNE_DIRS=(
  ".git"
  ".venv"
  "venv"
  "__pycache__"
  ".mypy_cache"
  ".pytest_cache"
  "node_modules"
  ".idea"
  ".vscode"
  "dist"
  "build"
  ".cache"
)

build_prune_expr() {
  local expr=""
  for d in "${PRUNE_DIRS[@]}"; do
    expr="$expr -name $d -o"
  done
  expr="${expr% -o}"
  echo "$expr"
}

PRUNE_EXPR="$(build_prune_expr)"

now="$(date '+%Y-%m-%d %H:%M:%S')"

{
  echo "# Project State Summary"
  echo
  echo "- Generated: $now"
  echo "- Root: \`$ROOT\`"
  echo
  echo "---"
  echo
  echo "## 1. Top-level contents"
  echo
  ls -lah
  echo
  echo "---"
  echo
  echo "## 2. Directory tree (depth 3)"
  echo
  if command -v tree >/dev/null 2>&1; then
    tree -a -L 3 -I '.git|.venv|venv|__pycache__|node_modules|build|dist|.cache|.idea|.vscode'
  else
    find . \
      \( $PRUNE_EXPR \) -prune -o \
      -maxdepth 3 -print | sort
  fi
  echo
  echo "---"
  echo
  echo "## 3. File inventory"
  echo
  echo "| File | Size(bytes) | Modified | Empty? | Type |"
  echo "|---|---:|---|---|---|"

  find . \
    \( $PRUNE_EXPR \) -prune -o \
    -type f -print | sort | head -n "$MAX_FILES_LIST" | while read -r f; do
      size="$(stat -c '%s' "$f" 2>/dev/null || echo '?')"
      mtime="$(stat -c '%y' "$f" 2>/dev/null | cut -d'.' -f1 || echo '?')"
      if [ "$size" = "0" ]; then
        empty="YES"
      else
        empty="NO"
      fi
      ftype="$(file -b "$f" 2>/dev/null | sed 's/|/\\|/g' || echo '?')"
      echo "| \`$f\` | $size | $mtime | $empty | $ftype |"
    done
  echo
  echo "> Note: inventory capped at first $MAX_FILES_LIST files."
  echo
  echo "---"
  echo
  echo "## 4. Likely important project files"
  echo
  find . \
    \( $PRUNE_EXPR \) -prune -o \
    -type f \( \
      -iname '*train*' -o \
      -iname '*data*' -o \
      -iname '*dataset*' -o \
      -iname '*loader*' -o \
      -iname '*client*' -o \
      -iname '*server*' -o \
      -iname '*fed*' -o \
      -iname '*prox*' -o \
      -iname '*scaffold*' -o \
      -iname '*dp*' -o \
      -iname '*privacy*' -o \
      -iname '*config*' -o \
      -iname 'requirements.txt' -o \
      -iname 'environment.yml' -o \
      -iname 'README*' \
    \) -print | sort
  echo
  echo "---"
  echo
  echo "## 5. Keyword scan"
  echo
} > "$OUT"

KEYWORDS=(
  "FedAvg"
  "FedProx"
  "SCAFFOLD"
  "scaffold"
  "federated"
  "aggregation"
  "aggregate"
  "server"
  "client"
  "round"
  "num_clients"
  "participation"
  "local_epochs"
  "centralized"
  "local-only"
  "macro-f1"
  "confusion"
  "roc"
  "auc"
  "opacus"
  "privacy"
  "epsilon"
  "delta"
  "clip"
  "noise"
  "ResNet"
  "ISIC"
)

for kw in "${KEYWORDS[@]}"; do
  {
    echo
    echo "### Keyword: \`$kw\`"
    matches="$(find . \
      \( $PRUNE_EXPR \) -prune -o \
      -type f -regextype posix-extended -iregex ".*$TEXT_EXT_REGEX" -print \
      | xargs -r grep -nHi -- "$kw" 2>/dev/null | head -n 20)"
    if [ -n "$matches" ]; then
      echo
      echo '```text'
      echo "$matches"
      echo '```'
    else
      echo
      echo "_No matches found._"
    fi
  } >> "$OUT"
done

{
  echo
  echo "---"
  echo
  echo "## 6. Non-empty text/code files with preview"
  echo
} >> "$OUT"

find . \
  \( $PRUNE_EXPR \) -prune -o \
  -type f -regextype posix-extended -iregex ".*$TEXT_EXT_REGEX" -print | sort | while read -r f; do
    size="$(stat -c '%s' "$f" 2>/dev/null || echo 0)"
    [ "$size" -eq 0 ] && continue

    line_count="$(wc -l < "$f" 2>/dev/null || echo 0)"
    {
      echo
      echo "### \`$f\`"
      echo "- Size: $size bytes"
      echo "- Lines: $line_count"
      echo "- Modified: $(stat -c '%y' "$f" 2>/dev/null | cut -d'.' -f1)"
      echo
      echo "Preview:"
      echo '```text'
      head -n "$MAX_TEXT_PREVIEW" "$f" 2>/dev/null
      echo '```'
    } >> "$OUT"
done

{
  echo
  echo "---"
  echo
  echo "## 7. Results and artifacts"
  echo
  echo
  echo "### Images / plots"
  find . \
    \( $PRUNE_EXPR \) -prune -o \
    -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.pdf' -o -iname '*.svg' \) -print | sort
  echo
  echo "### Model checkpoints"
  find . \
    \( $PRUNE_EXPR \) -prune -o \
    -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.bin' -o -iname '*.onnx' \) -print | sort
  echo
  echo "### Structured outputs"
  find . \
    \( $PRUNE_EXPR \) -prune -o \
    -type f \( -iname '*.csv' -o -iname '*.json' -o -iname '*.yaml' -o -iname '*.yml' -o -iname '*.log' \) -print | sort
  echo
  echo "---"
  echo
  echo "## 8. Simple stage estimate"
} >> "$OUT"

has_pattern() {
  find . \
    \( $PRUNE_EXPR \) -prune -o \
    -type f -regextype posix-extended -iregex ".*$TEXT_EXT_REGEX" -print \
    | xargs -r grep -Rqi -- "$1" 2>/dev/null
}

has_file_glob() {
  find . \
    \( $PRUNE_EXPR \) -prune -o \
    -type f \( $1 \) -print -quit 2>/dev/null | grep -q .
}

week_guess="Unknown"
reason="Not enough signals found."

if has_pattern 'opacus\|epsilon\|delta\|privacy'; then
  week_guess="Week 9+"
  reason="Differential privacy keywords found."
elif has_pattern 'SCAFFOLD\|server_control\|client_control\|control variate'; then
  week_guess="Week 8"
  reason="SCAFFOLD-related signals found."
elif has_pattern 'FedProx\|proximal\|mu'; then
  week_guess="Week 7"
  reason="FedProx-related signals found."
elif has_pattern 'FedAvg\|aggregation\|aggregate\|round\|num_clients'; then
  week_guess="Week 6"
  reason="Federated training / round-based signals found."
elif has_pattern 'centralized' && has_pattern 'client'; then
  week_guess="Week 5"
  reason="Centralized and per-client/local signals found."
elif has_pattern 'centralized\|confusion\|macro-f1'; then
  week_guess="Week 4"
  reason="Centralized baseline signals found."
elif has_pattern 'entropy\|distribution\|heatmap\|Jensen\|Shannon'; then
  week_guess="Week 3"
  reason="Heterogeneity-analysis signals found."
elif has_pattern 'dataset\|dataloader\|preprocess\|augmentation'; then
  week_guess="Week 2"
  reason="Dataset pipeline signals found."
elif has_pattern 'requirements.txt\|environment.yml\|config'; then
  week_guess="Week 1"
  reason="Environment/repo setup signals found."
fi

{
  echo
  echo "- Estimated stage: **$week_guess**"
  echo "- Basis: $reason"
  echo
  echo "---"
  echo
  echo "## 9. Suggested manual checks"
  echo
  echo "1. Open files under \`src/\`, \`scripts/\`, and \`configs/\`."
  echo "2. Check whether there are round-based logs or only centralized training."
  echo "3. Check whether model checkpoints exist for local clients or federated rounds."
  echo "4. Check the newest modified files first."
} >> "$OUT"

echo "Created $OUT"
