#!/usr/bin/env bash
set -Eeuo pipefail

# Monitored Unsloth training script with S3 progress tracking (hardened)
# Drop-in replacement for run_unsloth_training_monitored.sh
# Usage:
#   ./run_unsloth_training_monitored_fixed.sh            # defaults to 'remote' if env is set, else 'local'
#   ./run_unsloth_training_monitored_fixed.sh local
#   ./run_unsloth_training_monitored_fixed.sh remote

echo "=== Monitored Unsloth Qwen3-14B Training Script (fixed) ==="
echo "Starting at: $(date)"

# ---------- helpers ----------
die() { echo "ERROR: $*" >&2; exit 1; }
notice() { echo "» $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"; }

# Determine script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load environment variables
if [ -f "$SCRIPT_DIR/../../.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../../.env"
  notice "Loaded environment from ../../.env"
elif [ -f "$SCRIPT_DIR/.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  notice "Loaded environment from ./.env"
elif [ -f ".env" ]; then
  # shellcheck disable=SC1091
  source ".env"
  notice "Loaded environment from current ./.env"
else
  notice "No .env found; proceeding with defaults"
fi

# ---------- configuration ----------
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-14B-unsloth-bnb-4bit}"
OUTPUT_DIR="${OUTPUT_DIR:-qwen3_14b_iac_verbosity_sft}"
S3_MODEL_PATH="${S3_MODEL_PATH:-s3://asoba-llm-cache/models/Qwen/Qwen3-14B-unsloth-bnb-4bit/}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/home/ubuntu/Qwen3-14B-unsloth-bnb-4bit}"
REMOTE_USER="${QWEN_GPU_INSTANCE_USER:-ubuntu}"
REMOTE_HOST="${QWEN_GPU_INSTANCE_IP:-}"
SSH_KEY="${SSH_KEY_PATH:-}"

S3_BUCKET="${S3_BUCKET:-asoba-llm-cache}"
RUN_ID="qwen3-14b-$(date +%Y%m%d-%H%M%S)"
MONITOR_DIR="/tmp/training_monitor_${RUN_ID}"
mkdir -p "$MONITOR_DIR"

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# Monitoring scripts discovery (prefer <script_dir>/monitoring)
if [ -f "$SCRIPT_DIR/monitoring/training_monitor.py" ]; then
  MONITOR_PATH="$SCRIPT_DIR/monitoring"
elif [ -f "$SCRIPT_DIR/../monitoring/training_monitor.py" ]; then
  MONITOR_PATH="$SCRIPT_DIR/../monitoring"
elif [ -f "./monitoring/training_monitor.py" ]; then
  MONITOR_PATH="$(pwd)/monitoring"
else
  MONITOR_PATH=""
fi
MONITOR_SCRIPT="${MONITOR_PATH:+$MONITOR_PATH/training_monitor.py}"

# ---------- validations ----------
require_cmd python3
require_cmd aws

check_aws() {
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    die "AWS credentials not configured. Run 'aws configure' or set env vars."
  fi
}
check_gpu() {
  require_cmd nvidia-smi
  notice "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
}
detect_torch_index() {
  # Default to CUDA 12.1 wheels when possible, else 11.8, else CPU wheels as a fallback
  local cuda_ver
  if cuda_ver="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1)"; then
    cuda_ver="${cuda_ver%%.*}.${cuda_ver#*.}"
    case "$cuda_ver" in
      12.*) echo "https://download.pytorch.org/whl/cu121" ;;
      11.*) echo "https://download.pytorch.org/whl/cu118" ;;
      *) echo "https://download.pytorch.org/whl/cu121" ;;
    esac
  else
    echo "https://download.pytorch.org/whl/cpu"
  fi
}

# ---------- dependency setup ----------
install_python_deps() {
  notice "Installing Python deps…"
  python3 -m pip install --upgrade pip wheel setuptools >/dev/null

  local TORCH_INDEX
  TORCH_INDEX="$(detect_torch_index)"
  if ! python3 -m pip install -q torch torchvision torchaudio --index-url "$TORCH_INDEX"; then
    notice "Torch install from $TORCH_INDEX failed; falling back to CPU wheels"
    python3 -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi

  python3 -m pip install -q "transformers>=4.41.0" "accelerate>=0.30.0" "peft>=0.11.0" bitsandbytes datasets >/dev/null 2>&1 || true

  # Unsloth and TRL
  if ! python3 -m pip install -q unsloth; then
    notice "PyPI unsloth failed; trying GitHub main"
    python3 -m pip install -q "unsloth @ git+https://github.com/unslothai/unsloth.git"
  fi
  python3 -m pip install -q trl >/dev/null 2>&1 || true
  notice "Deps done."
}

# ---------- monitoring ----------
start_sync_loop() {
  # Simple, robust S3 sync loop independent of Python background threads
  notice "Starting S3 sync loop to s3://$S3_BUCKET/logs/training-runs/$RUN_ID/ (every 30s)"
  (
    set -Eeuo pipefail
    while [ ! -f "$MONITOR_DIR/_stop_sync" ]; do
      aws s3 sync "$MONITOR_DIR" "s3://$S3_BUCKET/logs/training-runs/$RUN_ID/" --region "$AWS_DEFAULT_REGION" >/dev/null 2>&1 || true
      sleep 30
    done
    aws s3 sync "$MONITOR_DIR" "s3://$S3_BUCKET/logs/training-runs/$RUN_ID/" --region "$AWS_DEFAULT_REGION" >/dev/null 2>&1 || true
  ) &
  echo $! > "$MONITOR_DIR/sync.pid"
}

setup_monitoring() {
  notice "Setting up monitoring at $MONITOR_DIR"
  if [ -n "$MONITOR_SCRIPT" ] && [ -f "$MONITOR_SCRIPT" ]; then
    python3 - <<PY || true
import sys, os
sys.path.append("$MONITOR_PATH")
try:
    from training_monitor import create_run_metadata
    create_run_metadata("$RUN_ID", "$BASE_MODEL", "$MONITOR_DIR")
    print("Metadata created via training_monitor.py")
except Exception as e:
    print(f"training_monitor.py not available/failed: {e}")
PY
  fi
  start_sync_loop
  echo ""
  echo "🔍 To monitor progress from another terminal:"
  if [ -n "$MONITOR_PATH" ]; then
    echo "   python3 \"$MONITOR_PATH/monitor_training.py\" --run-id \"$RUN_ID\""
  else
    echo "   tail -f \"$MONITOR_DIR/training.log\""
  fi
  echo ""
}

cleanup_monitoring() {
  notice "Cleaning up monitoring…"
  touch "$MONITOR_DIR/_stop_sync"
  if [ -f "$MONITOR_DIR/sync.pid" ]; then
    SYNC_PID="$(cat "$MONITOR_DIR/sync.pid" || true)"
    if [ -n "${SYNC_PID:-}" ]; then
      kill "$SYNC_PID" >/dev/null 2>&1 || true
      wait "$SYNC_PID" >/dev/null 2>&1 || true
    fi
  fi
  echo "Training completed at: $(date)" > "$MONITOR_DIR/_complete"
  aws s3 sync "$MONITOR_DIR" "s3://$S3_BUCKET/logs/training-runs/$RUN_ID/" --region "$AWS_DEFAULT_REGION" >/dev/null 2>&1 || true
  notice "Monitoring cleanup complete."
}

update_progress_json() {
  # Best-effort: use training_monitor.parse_training_log if present
  if [ -f "$MONITOR_DIR/training.log" ] && [ -n "$MONITOR_SCRIPT" ] && [ -f "$MONITOR_SCRIPT" ]; then
    python3 - <<PY || true
import sys, json
sys.path.append("$MONITOR_PATH")
try:
    from training_monitor import parse_training_log
    with open("$MONITOR_DIR/training.log","r") as f: content = f.read()
    progress = parse_training_log(content)
    with open("$MONITOR_DIR/progress.json","w") as f: json.dump(progress, f, indent=2)
except Exception as e:
    pass
PY
  fi
}

# ---------- artifacts ----------
download_model_from_s3() {
  if [ -d "$LOCAL_MODEL_PATH" ]; then
    notice "Base model present at $LOCAL_MODEL_PATH"
  else
    notice "Downloading base model from $S3_MODEL_PATH → $LOCAL_MODEL_PATH"
    mkdir -p "$LOCAL_MODEL_PATH"
    aws s3 sync "$S3_MODEL_PATH" "$LOCAL_MODEL_PATH" --region "$AWS_DEFAULT_REGION"
  fi
}

upload_results_to_s3() {
  notice "Uploading results…"
  local uploader
  if [ -n "$MONITOR_PATH" ] && [ -f "$MONITOR_PATH/s3_model_uploader.py" ]; then
    uploader="$MONITOR_PATH/s3_model_uploader.py"
    python3 "$uploader" "$OUTPUT_DIR" --run-id "$RUN_ID" --monitoring-dir "$MONITOR_DIR" --bucket "$S3_BUCKET" --region "$AWS_DEFAULT_REGION" && {
      if [ -f "$OUTPUT_DIR/s3_upload_manifest.json" ]; then
        local s3p
        s3p="$(python3 -c "import json,sys; print(json.load(open('$OUTPUT_DIR/s3_upload_manifest.json'))['s3_path'])" || true)"
        [ -n "$s3p" ] && echo "📍 Model location: $s3p"
      fi
      echo "✅ Model successfully uploaded to S3"
      return 0
    }
    echo "⚠️ s3_model_uploader.py failed, falling back to raw sync"
  fi
  aws s3 cp --recursive "$OUTPUT_DIR" "s3://$S3_BUCKET/models/training-runs/$RUN_ID/$OUTPUT_DIR/" --region "$AWS_DEFAULT_REGION"
  echo "📍 Model location: s3://$S3_BUCKET/models/training-runs/$RUN_ID/$OUTPUT_DIR/"
}

# ---------- local run ----------
run_local() {
  trap cleanup_monitoring EXIT INT TERM

  echo ""
  echo "=== Step 1: GPU check ==="
  check_gpu

  echo ""
  echo "=== Step 2: AWS check ==="
  check_aws

  echo ""
  echo "=== Step 3: Monitoring setup ==="
  setup_monitoring

  echo ""
  echo "=== Step 4: Python deps ==="
  install_python_deps

  echo ""
  echo "=== Step 5: Base model ==="
  download_model_from_s3

  echo ""
  echo "=== Step 6: Verify data ==="
  VERBOSITY_JSONL="${VERBOSITY_JSONL:-verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl}"
  IAC_JSONL="${IAC_JSONL:-data/final_enhanced_iac_corpus.jsonl}"
  [ -f "$VERBOSITY_JSONL" ] || die "$VERBOSITY_JSONL not found"
  [ -f "$IAC_JSONL" ] || die "$IAC_JSONL not found"
  echo "✓ Found $VERBOSITY_JSONL"
  echo "✓ Found $IAC_JSONL"

  echo ""
  echo "=== Step 7: Training (this may take hours) ==="
  echo "🔍 Monitor: $([ -n "$MONITOR_PATH" ] && echo "python3 \"$MONITOR_PATH/monitor_training.py\" --run-id \"$RUN_ID\"" || echo "tail -f \"$MONITOR_DIR/training.log\"")"
  echo ""

  # line-buffered output so tee writes promptly
  if command -v stdbuf >/dev/null 2>&1; then
    STD_BUF="stdbuf -oL -eL"
  else
    STD_BUF=""
  fi

  set +o pipefail
  $STD_BUF python3 "$SCRIPT_DIR/train_qwen3_14b_optimal_sft" \
    --base_model "$BASE_MODEL" \
    --local_jsonl "$VERBOSITY_JSONL:0.35" \
    --local_jsonl "$IAC_JSONL:0.50" \
    --add_openmath_reasoning 0.10 \
    --add_finetome_chat 0.05 \
    --output_dir "$OUTPUT_DIR" \
    --max_len 4096 \
    --epochs 2 \
    2>&1 | tee -a "$MONITOR_DIR/training.log" | while IFS= read -r _; do update_progress_json; done
  status=${PIPESTATUS[0]}
  set -o pipefail
  [ "$status" -eq 0 ] || die "Training failed (exit $status)"

  echo ""
  echo "=== Step 8: Upload artifacts ==="
  if [ -d "$OUTPUT_DIR" ]; then
    upload_results_to_s3 || true
  else
    echo "WARNING: $OUTPUT_DIR not found; skipping upload"
  fi

  echo ""
  echo "=== Done ==="
  echo "Finished at: $(date)"
  echo "Run ID: $RUN_ID"
  echo "Logs: s3://$S3_BUCKET/logs/training-runs/$RUN_ID/"
}

# ---------- remote run ----------
deploy_and_run_remote() {
  [ -n "$REMOTE_HOST" ] || die "QWEN_GPU_INSTANCE_IP not set (REMOTE_HOST)"
  [ -n "$SSH_KEY" ] || die "SSH_KEY_PATH not set"
  [ -f "$SSH_KEY" ] || die "SSH key not found at $SSH_KEY"
  require_cmd ssh
  require_cmd scp

  notice "Deploying to $REMOTE_USER@$REMOTE_HOST"
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "mkdir -p ~/qwen_training"
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$0" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/"
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SCRIPT_DIR/train_qwen3_14b_optimal_sft" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/"
  if [ -n "$MONITOR_PATH" ] && [ -d "$MONITOR_PATH" ]; then
    scp -r -i "$SSH_KEY" -o StrictHostKeyChecking=no "$MONITOR_PATH" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/"
  fi
  # Try best-effort to provide a .env in remote workdir
  if [ -f "$SCRIPT_DIR/../../.env" ]; then
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SCRIPT_DIR/../../.env" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/.env" || true
  elif [ -f "$SCRIPT_DIR/.env" ]; then
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SCRIPT_DIR/.env" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/.env" || true
  elif [ -f ".env" ]; then
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no ".env" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/.env" || true
  fi

  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" \
    "cd ~/qwen_training && chmod +x ./$(basename "$0") && ./$(basename "$0") local"
}

# ---------- main ----------
MODE="${1:-auto}"
case "$MODE" in
  local)   run_local ;;
  remote)  deploy_and_run_remote ;;
  auto)
    if [ -n "${REMOTE_HOST:-}" ] && [ -n "${SSH_KEY:-}" ]; then
      notice "Remote variables detected; using remote mode"
      deploy_and_run_remote
    else
      notice "No remote variables; running locally"
      run_local
    fi
    ;;
  *)
    die "Unknown mode '$MODE' (use: local | remote)"
    ;;
esac
