#!/usr/bin/env bash
# resolve_model.sh — syncs a model directory from S3 to local disk deterministically
#
# Usage:
#   resolve_model.sh --tag mistral-7b-v0.3 [--bucket asoba-llm-cache] [--region us-west-2] [--local-root /mnt/training/models]
#
# Exit codes:
#   0  success (model ready)
#   1  no source found
#   2  sync failed
#   3  verification failed

exec > /tmp/resolve_model_output.log 2>&1
set -euo pipefail
set -x

# ---- arg parsing ------------------------------------------------------------

export TAG=""
BUCKET="asoba-llm-cache"
REGION="$(aws configure get region || echo us-east-1)"
LOCAL_ROOT="/mnt/training/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) export TAG="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --local-root) LOCAL_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 64 ;;
  esac
done

[[ -z "$TAG" ]] && { echo "[ERR] --tag is required" >&2; exit 64; }

LOCAL_DIR="${LOCAL_ROOT}/${TAG}"
TRAINED_PREFIX="s3://${BUCKET}/trained-models/${TAG}/"
BASE_PREFIX="s3://${BUCKET}/models/${TAG}/"

echo "[INFO] Resolving model '${TAG}' from bucket '${BUCKET}' (region ${REGION})"

# ---- source resolution ------------------------------------------------------

SOURCE=""
if aws s3 ls "${TRAINED_PREFIX}" --region "${REGION}" >/dev/null 2>&1; then
  SOURCE="${TRAINED_PREFIX}"
  echo "[INFO] Found trained model at ${SOURCE}"
elif aws s3 ls "${BASE_PREFIX}" --region "${REGION}" >/dev/null 2>&1; then
  SOURCE="${BASE_PREFIX}"
  echo "[INFO] Falling back to base model at ${SOURCE}"
else
  echo "[ERR] No model found under ${TRAINED_PREFIX} or ${BASE_PREFIX}" >&2
  exit 1
fi

# ---- sync -------------------------------------------------------------------

mkdir -p "${LOCAL_DIR}"
echo "[INFO] Syncing model to ${LOCAL_DIR}"
if ! aws s3 sync "${SOURCE}" "${LOCAL_DIR}/" --region "${REGION}" --size-only; then
  echo "[ERR] Sync failed" >&2
  exit 2
fi

# ---- verification -----------------------------------------------------------

CONFIG="${LOCAL_DIR}/config.json"
WEIGHTS_COUNT=$(find "${LOCAL_DIR}" -maxdepth 1 -type f \( -name "*.bin" -o -name "*.safetensors" \) | wc -l)

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERR] config.json missing in ${LOCAL_DIR}" >&2
  exit 3
fi
if [[ "${WEIGHTS_COUNT}" -eq 0 ]]; then
  echo "[ERR] No weight files (.bin/.safetensors) found in ${LOCAL_DIR}" >&2
  exit 3
fi

# ---- provenance -------------------------------------------------------------

echo "${SOURCE}" > "${LOCAL_DIR}/.source_s3"

cat > /tmp/gen_manifest.py <<'EOF'
import hashlib, json, os, sys
cfg = sys.argv[1]
local_dir = os.path.dirname(cfg)
sha = hashlib.sha256(open(cfg,'rb').read()).hexdigest()
manifest = {
  "tag": os.environ["TAG"],
  "source_s3": open(os.path.join(local_dir,".source_s3")).read().strip(),
  "config_sha256": sha
}
with open(os.path.join(local_dir,'.manifest.json'),'w') as f:
  json.dump(manifest,f,indent=2)
print(f"[INFO] Wrote provenance to {local_dir}/.manifest.json")
EOF

python3 /tmp/gen_manifest.py "${CONFIG}"
rm -f /tmp/gen_manifest.py

echo "[OK] Model '${TAG}' ready at ${LOCAL_DIR}"
exit 0