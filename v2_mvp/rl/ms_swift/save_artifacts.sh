#!/usr/bin/env bash
set -euo pipefail

cd /Users/apple/2N-Guardian

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="v2_mvp/rl/artifacts"
OUT_TAR="$OUT_DIR/rl_pack_${STAMP}.tar.gz"
MANIFEST="$OUT_DIR/rl_pack_${STAMP}.manifest.txt"

mkdir -p "$OUT_DIR"

FILES=(
  "v2_mvp/rl/data/2n_rl_data_clean_v1.jsonl"
  "v2_mvp/rl/data/eval_set.jsonl"
  "v2_mvp/rl/data/freeze_manifest.json"
  "v2_mvp/rl/data/law_enrich_report_v2.json"
  "v2_mvp/rl/reward_manager.py"
  "v2_mvp/rl/ms_swift/dataset_info.json"
  "v2_mvp/rl/ms_swift/grpo_config.yaml"
  "v2_mvp/rl/ms_swift/run_train.sh"
  "v2_mvp/rl/ms_swift/reward_swift_adapter.py"
  "v2_mvp/rl/ms_swift/dry_run_reward.py"
  "v2_mvp/rl/ms_swift/preflight_gate.py"
  "v2_mvp/rl/docs/reward_spec.md"
  "v2_mvp/rl/docs/legal_verifier_prompt.txt"
)

# Include project reward dependencies under src/.
SRC_FILES=()
while IFS= read -r f; do
  SRC_FILES+=("$f")
done < <(find src -type f \
  ! -path "*/__pycache__/*" \
  ! -name "*.pyc" \
  | sort)
FILES+=( "${SRC_FILES[@]}" )

for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] missing $f"
    exit 1
  fi
done

tar -czf "$OUT_TAR" "${FILES[@]}"

echo "artifact: $OUT_TAR" > "$MANIFEST"
for f in "${FILES[@]}"; do
  shasum -a 256 "$f" >> "$MANIFEST"
done

echo "saved: $OUT_TAR"
echo "manifest: $MANIFEST"
