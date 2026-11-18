#!/bin/bash
# Download the AlphaFree datasets (+ the official AlphaFree pre-trained weights) into LRRec/data.
#
# Adapted from the official AlphaFree script (github.com/Minseojeonn/AlphaFree, data/download.sh).
# Changes: DATA_DIR defaults to <repo>/data (a symlink to large storage), the zip is staged in
# DATA_DIR/.tmp instead of /tmp, and gdown falls back to `uvx gdown` instead of `pip install --user`.
# The on-disk format is left exactly as AlphaFree ships it:
#   data/<dataset>/cf_data/{train,valid,test}.txt
#   data/<dataset>/item_info/item_cf_embeds_{large3,LLAMA}_array.npy, item_title.json, ...
#   data/pretrained/<dataset>.pth.tar
#
# Usage:
#   scripts/download_data.sh                         # all 7 datasets
#   scripts/download_data.sh --dataset Movie,Video   # selected only
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
WEIGHT_DIR="$DATA_DIR/pretrained"
STAGE_DIR="$DATA_DIR/.tmp"
mkdir -p "$DATA_DIR" "$WEIGHT_DIR" "$STAGE_DIR"

if command -v gdown &> /dev/null; then
  GDOWN=(gdown)
elif command -v uvx &> /dev/null; then
  GDOWN=(uvx gdown)
else
  echo "Error: neither gdown nor uvx found. Install gdown first."
  exit 1
fi
if ! command -v unzip &> /dev/null; then
  echo "Error: unzip not found. Please install unzip first."
  exit 1
fi

# Google Drive ids (from the official script)
declare -A DATA_ZIPS
DATA_ZIPS["Steam"]="1abykeNcnxAJpqZB1mYM9bg9MI3pehSN4"
DATA_ZIPS["Movie"]="1Wpl3xJUlVhPPyvbWX81e8RLfaYFBOk8_"
DATA_ZIPS["Book"]="1FME3AMNpk7EMgmGiFOZa67AKupR-5QuF"
DATA_ZIPS["Video"]="1fslBxD3kcP58eylGAsjRoy9cOOn1S7uI"
DATA_ZIPS["Baby"]="15KQtrujfR-SjMU4NzVOCSSayNfmtJjDu"
DATA_ZIPS["Beauty"]="1sVMNdBgdK7uAdAqY8c1rc9ds9ywyFhX9"
DATA_ZIPS["Health"]="14F5QSXSNKrpnTeKPx757Xx04j1jvt-V7"

declare -A WEIGHTS
WEIGHTS["Movie"]="125c6BeG-AVPaFO519aW_PkDni9Dp6-IZ"
WEIGHTS["Book"]="1Y18ZFq9yMscdl3Sc395SlnF37dBAKlUz"
WEIGHTS["Video"]="1k-Mi3OjadXL4zL6_aN2UjhPUPwMHH7tj"
WEIGHTS["Baby"]="1E8-cZILoU-5o62kY5edBV84ZkzK3A3IE"
WEIGHTS["Steam"]="1sEh_WbFm5mA7gTBtFuvkS5AmkPvRDpdr"
WEIGHTS["Beauty"]="1bNRKzE_s4SXpDF3qWl7ox4YIXlpcw1M7"
WEIGHTS["Health"]="1LScqWmIu7ZB7tByxG_OB3UfNMsvmU5u3"

declare -A CANON
CANON["Steam"]="steam"
CANON["Movie"]="amazon_movie"
CANON["Book"]="amazon_book_2014"
CANON["Video"]="amazon_video"
CANON["Baby"]="amazon_baby"
CANON["Beauty"]="amazon_beauty_personal"
CANON["Health"]="amazon_health"

ALL_DATASETS=(Steam Movie Book Video Baby Beauty Health)

SELECTED_DATASETS=()
JOBS=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) shift ;;
    --dataset) IFS=',' read -r -a SELECTED_DATASETS <<< "${2:-}"; shift 2 ;;
    --jobs) JOBS="${2:-3}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--dataset A,B] [--jobs N]   (datasets: ${ALL_DATASETS[*]})"; exit 0 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

download_weight() {
  local NAME="$1"
  local FILE_ID="${WEIGHTS[$NAME]:-}"
  local OUT_PATH="${WEIGHT_DIR}/${CANON[$NAME]}.pth.tar"
  if [[ -z "$FILE_ID" ]]; then echo "[WARN] no weight id for $NAME"; return 0; fi
  if [[ -f "$OUT_PATH" ]]; then echo "[SKIP] weight exists: $OUT_PATH"; return 0; fi
  echo "  -> (W) $NAME"
  "${GDOWN[@]}" "$FILE_ID" -O "$OUT_PATH.part"
  mv "$OUT_PATH.part" "$OUT_PATH"
}

download_dataset() {
  local NAME="$1"
  local FILE_ID="${DATA_ZIPS[$NAME]:-}"
  local CAN="${CANON[$NAME]:-}"
  local OUT_DIR="${DATA_DIR}/${CAN}"
  if [[ -z "$FILE_ID" || -z "$CAN" ]]; then echo "Error: unknown dataset '$NAME'"; exit 1; fi
  if [[ -d "$OUT_DIR" ]]; then echo "[SKIP] dataset exists: $OUT_DIR"; return 0; fi

  local TMP_DIR
  TMP_DIR="$(mktemp -d -p "$STAGE_DIR")"
  echo "  -> (D) downloading $NAME"
  "${GDOWN[@]}" "$FILE_ID" -O "$TMP_DIR/$NAME.zip"
  echo "  -> (D) extracting $NAME"
  unzip -o "$TMP_DIR/$NAME.zip" -d "$TMP_DIR/extracted" > /dev/null
  local TOP_DIR
  TOP_DIR="$(find "$TMP_DIR/extracted" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
  if [[ -z "$TOP_DIR" ]]; then echo "Error: cannot find extracted folder for $NAME"; rm -rf "$TMP_DIR"; exit 1; fi
  mv "$TOP_DIR" "$OUT_DIR"
  rm -rf "$TMP_DIR"
  echo "  -> (D) ready: $OUT_DIR"
}

download_both() {
  local NAME="$1"
  echo "[DATA+WEIGHT] $NAME"
  local rc=0
  download_dataset "$NAME" & local pid_d=$!
  download_weight "$NAME" & local pid_w=$!
  wait "$pid_d" || rc=1
  wait "$pid_w" || rc=1
  if [[ $rc -ne 0 ]]; then echo "[FAIL] $NAME"; return 1; fi
  echo "[DONE] $NAME"
}

TARGETS=("${ALL_DATASETS[@]}")
if [[ ${#SELECTED_DATASETS[@]} -gt 0 ]]; then TARGETS=("${SELECTED_DATASETS[@]}"); fi
echo "Data dir: $DATA_DIR (-> $(readlink -f "$DATA_DIR"))"
echo "Targets: ${TARGETS[*]}  (parallel jobs: $JOBS)"

PIDS=()
for NAME in "${TARGETS[@]}"; do
  while [[ $(jobs -pr | wc -l) -ge "$JOBS" ]]; do sleep 1; done
  download_both "$NAME" &
  PIDS+=($!)
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done
rmdir "$STAGE_DIR" 2>/dev/null || true
if [[ $FAILED -ne 0 ]]; then echo "$FAILED dataset(s) failed."; exit 1; fi
echo "Completed (data + weights)."
