#!/bin/bash
# Evaluation: inspect -> visualize
#
# Produces under OUT_DIR:
#   metrics.txt  dataset summary (up to 200 sampled clips) + per-clip metrics for sanity clips
#   viz/         visualization panels for the sanity clips listed in the config
#
# Usage:
#   bash tools/run_eval.sh --config configs/mevis.yaml --input /data/mevis --output /events --out-dir eval/mevis
#
# --input / --output default to source_root / output_root from the config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=""
SOURCE_ROOT=""
OUTPUT_ROOT=""
OUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2";      shift 2 ;;
        --input)    SOURCE_ROOT="$2"; shift 2 ;;
        --output)   OUTPUT_ROOT="$2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2";     shift 2 ;;
        *) echo "[run_eval] Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "[run_eval] ERROR: --config required"; exit 1
fi

cfg() { python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('$1') or '$2')"; }
DATASET=$(cfg dataset "")
MODE=$(cfg mode frames)
SOURCE_ROOT="${SOURCE_ROOT:-$(cfg source_root "")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(cfg output_root "")}"
if [[ "$MODE" == "images" ]]; then METHOD="saccade+linear"; else METHOD="$(cfg upsample "")+linear"; fi
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "[run_eval] ERROR: no events root (pass --output or set output_root in the config)"; exit 1
fi

OUT_DIR="${OUT_DIR:-eval/${DATASET}/${METHOD}_${TIMESTAMP}}"
VIZ_DIR="${OUT_DIR}/viz"
METRICS_TXT="${OUT_DIR}/metrics.txt"

mkdir -p "$VIZ_DIR"

echo ""
echo "======================================================================"
echo "  EVAL  dataset=${DATASET}  method=${METHOD}"
echo "  output -> ${OUT_DIR}"
echo "======================================================================"

# Read sanity clips from config (while-read instead of mapfile: works with bash 3.2 on macOS)
SANITY_CLIPS=()
while IFS= read -r clip; do
    [[ -n "$clip" ]] && SANITY_CLIPS+=("$clip")
done < <(python3 -c "
import yaml; c = yaml.safe_load(open('$CONFIG'))
for clip in c.get('sanity_clips') or []:
    print(str(clip).strip())
")

# ── Step 1: dataset summary + sanity clip metrics -> metrics.txt ─────────────
echo ""
echo "[1/2] metrics -> ${METRICS_TXT}"
{
    echo "======================================================================"
    echo "  dataset=${DATASET}  method=${METHOD}  timestamp=${TIMESTAMP}"
    echo "======================================================================"
    echo ""

    python3 "${SCRIPT_DIR}/inspect_events.py" \
        --config "$CONFIG" \
        --output-root "$OUTPUT_ROOT" \
        --samples 200 \
        -j 8 \
        --verbose

} 2>&1 | tee "$METRICS_TXT"
echo "[1/2] done -> ${METRICS_TXT}"

# ── Step 2: visualize sanity clips ────────────────────────────────────────────
echo ""
echo "[2/2] visualize ${#SANITY_CLIPS[@]} sanity clips -> ${VIZ_DIR}/"
for CLIP_REL in ${SANITY_CLIPS[@]+"${SANITY_CLIPS[@]}"}; do
    CLIP_ID="${CLIP_REL##*/}"
    EVENTS_H5=$(find "${OUTPUT_ROOT}/${DATASET}" -path "*/${CLIP_ID}/events.h5" | head -1)

    if [[ -z "$EVENTS_H5" ]]; then
        echo "  [SKIP] events.h5 not found for clip: ${CLIP_ID}"
        continue
    fi

    if [[ "$MODE" == "frames" ]]; then
        # The sanity clip entry is a path to the clip dir relative to the input root
        SOURCE_PATH="${SOURCE_ROOT}/${CLIP_REL}"
    else
        SOURCE_PATH=$(find "$SOURCE_ROOT" -name "${CLIP_ID}.*" | head -1)
    fi

    if [[ -z "${SOURCE_PATH:-}" || ! -e "$SOURCE_PATH" ]]; then
        echo "  [SKIP] source not found for ${CLIP_ID}"
        continue
    fi

    VIZ_OUT="${VIZ_DIR}/${CLIP_ID}.png"
    echo "  ${CLIP_ID} -> ${VIZ_OUT}"
    python3 "${SCRIPT_DIR}/visualize.py" \
        "$SOURCE_PATH" "$EVENTS_H5" \
        --output "$VIZ_OUT" 2>&1 || echo "  [WARN] visualize failed for ${CLIP_ID}"
done
echo "[2/2] done"

echo ""
echo "======================================================================"
echo "  EVAL COMPLETE"
echo "  metrics : ${METRICS_TXT}"
echo "  viz     : ${VIZ_DIR}/"
echo "======================================================================"
