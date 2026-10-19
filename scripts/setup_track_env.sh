#!/usr/bin/env bash
# Build the `lone-track` conda env used by lone_track/ and the tracker benchmark.
#
# A clone of `cyberbrick`, so torch 2.7.0+cu128 (the build the RTX 5090 needs) and
# lerobot stay exactly as training uses them; the trackers are added on top
# without letting pip touch torch:
#   - co-tracker  pip -e, no deps (it declares none)
#   - sam2        pip -e, no deps, SAM2_BUILD_CUDA=0 (skips the optional CUDA
#                 extension: nvcc + sm_120 is a needless risk, and the extension
#                 only post-processes masks)
#   - tapnet      pip -e, no deps -- its declared deps are the jax stack; the
#                 torch models (tapnet/torch, tapnet/tapnext, tapnextpp) need
#                 only einshape and dm-tree
# Repos are pinned so benchmark numbers are reproducible. Checkpoints go to
# outputs/track/ckpts (gitignored). Safe to re-run: every step skips work done.
#
#   bash scripts/setup_track_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV=lone-track
TP="$ROOT/third_party"
CK="$ROOT/outputs/track/ckpts"

COTRACKER_SHA=82e02e8029753ad4ef13cf06be7f4fc5facdda4d
SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4
TAPNET_SHA=730cda1c730877cfedbe01bf87fb1cadb78a565d

if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" --clone cyberbrick
fi
PY="$(conda run -n "$ENV" which python)"
PIP=("$PY" -m pip)

clone() {  # clone <github owner/repo> <sha>
  local dir="$TP/$(basename "$1")"
  [ -d "$dir/.git" ] || git clone -q "https://github.com/$1.git" "$dir"
  git -C "$dir" fetch -q --depth 1 origin "$2" 2>/dev/null || true
  git -C "$dir" checkout -q "$2"
}
mkdir -p "$TP" "$CK"
clone facebookresearch/co-tracker "$COTRACKER_SHA"
clone facebookresearch/sam2 "$SAM2_SHA"
clone google-deepmind/tapnet "$TAPNET_SHA"

TORCH_BEFORE="$("$PY" -c 'import torch; print(torch.__version__)')"
"${PIP[@]}" install -q --no-deps -e "$TP/co-tracker"
SAM2_BUILD_CUDA=0 "${PIP[@]}" install -q --no-deps --no-build-isolation -e "$TP/sam2"
"${PIP[@]}" install -q --no-deps -e "$TP/tapnet"
"${PIP[@]}" install -q "hydra-core>=1.3.2" "iopath>=0.1.10" einshape dm-tree
"${PIP[@]}" install -q --no-deps -e "$ROOT"
TORCH_AFTER="$("$PY" -c 'import torch; print(torch.__version__)')"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "torch changed from $TORCH_BEFORE to $TORCH_AFTER -- rebuild the env" >&2
  exit 1
fi

fetch() {  # fetch <url> [name]
  local out="$CK/${2:-$(basename "$1")}"
  [ -s "$out" ] || curl -fL --retry 3 -o "$out" "$1"
}
fetch https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth cotracker3_offline.pth
fetch https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth cotracker3_online.pth
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
fetch https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
fetch https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt
fetch https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt

"${PIP[@]}" check || echo "pip check reported conflicts (see above)" >&2
"$PY" - <<'EOF'
import torch, cotracker, sam2, tapnet.torch.tapir_model, tapnet.tapnextpp.votsp2026.model
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("lone-track env OK")
EOF
