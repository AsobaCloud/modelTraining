#!/usr/bin/env bash
# safe_kill_duplicates.sh
# Keeps the OLDEST matching training process, terminates newer duplicates.
# Default match: "train_mistral_simple.py"
# Requires: bash, ps, pgrep; optionally uses nvidia-smi to help choose.
set -euo pipefail

NAME_REGEX="train_mistral_simple.py"
FORCE=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [-n NAME_REGEX] [--force] [--dry-run]

Options:
  -n REGEX     Process match (default: ${NAME_REGEX})
  --force      Do not prompt for confirmation
  --dry-run    Show what would be killed, but do nothing

Strategy:
  1) Collect PIDs matching REGEX.
  2) Choose the 'keep' PID = the one with largest elapsed time (ETIMES).
     If nvidia-smi is available, prefer a running GPU consumer on ties.
  3) Send SIGTERM to others, wait up to 10s each, then SIGKILL if needed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) NAME_REGEX="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Collect session leaders only (SID==PID)
mapfile -t LEADERS < <(ps -eo pid,sid,etimes,cmd --no-headers \
  | awk -v r="$NAME_REGEX" '$4 ~ r && $1==$2 {print $1" "$3" "$4}')

if ((${#LEADERS[@]}<=1)); then
  echo "Leaders found: ${#LEADERS[@]}. Nothing to do."; exit 0
fi

# Optional: GPU usage map from nvidia-smi
declare -A GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  # pid,used_gpu_memory [MiB]
  while IFS=, read -r pid mem; do
    pid="$(echo "$pid" | xargs || true)"
    mem="$(echo "$mem" | tr -dc '0-9' || true)"
    [[ -n "$pid" && -n "$mem" ]] && GPU["$pid"]="$mem"
  done < <(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)
fi

# Choose keep = largest ETIMES
KEEP_PID=""
KEEP_ET=-1
for line in "${LEADERS[@]}"; do
  pid=$(awk '{print $1}' <<<"$line"); et=$(awk '{print $2}' <<<"$line")
  if (( et > KEEP_ET )); then KEEP_PID="$pid"; KEEP_ET="$et"; fi
done

echo "Keeping leader PID $KEEP_PID (etimes=$KEEP_ET)"
# Gather descendants of KEEP_PID (recursive)
descendants() {
  local p="$1"; shift || true
  local kids
  mapfile -t kids < <(pgrep -P "$p" || true)
  for k in "${kids[@]}"; do
    echo "$k"
    descendants "$k"
  done
}
mapfile -t KEEP_TREE < <(printf "%s\n" "$KEEP_PID"; descendants "$KEEP_PID" | sort -u)

# Build kill list = (all other leaders + their descendants) MINUS KEEP_TREE
KILL=()
for line in "${LEADERS[@]}"; do
  pid=$(awk '{print $1}' <<<"$line")
  [[ "$pid" == "$KEEP_PID" ]] && continue
  # descendants of this other leader:
  mapfile -t tree < <(printf "%s\n" "$pid"; descendants "$pid" | sort -u)
  for p in "${tree[@]}"; do
    # skip if in KEEP_TREE (never kill kept leader or its workers)
    if ! printf "%s\n" "${KEEP_TREE[@]}" | grep -qx "$p"; then
      KILL+=("$p")
    fi
  done
done

# Dedup
readarray -t KILL < <(printf "%s\n" "${KILL[@]}" | sort -u)
echo "Will terminate ${#KILL[@]} processes: ${KILL[*]}"

if (( DRY_RUN )); then
  echo "(dry-run) Exiting without changes."
  exit 0
fi

if (( ! FORCE )); then
  read -r -p "Proceed to kill others? Type 'yes' to continue: " ACK
  if [[ "$ACK" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

# TERM then KILL
for pid in "${KILL[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done
for pid in "${KILL[@]}"; do
  for _ in {1..10}; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null || true
done

echo "Done. Remaining PID: $KEEP_PID"