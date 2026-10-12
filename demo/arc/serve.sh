#!/bin/bash
# Serve a pi0.5 policy for demo/rollout.py. Run on an ARC GPU node (inside `interact ... --gres=gpu:1`),
# from the openpi folder:
#     bash <path>/serve.sh pnp 9999        # checkpoints/pnp/pnp_v1/9999 (fine-tuned, training config pnp)
#     bash <path>/serve.sh pnp             # list the saved steps of pnp
#     bash <path>/serve.sh base            # pi05_droid as is
# Third argument: experiment name (default <task>_v1). PORT=9000 by default.
task=$1; step=$2; exp=${3:-${task}_v1}; port=${PORT:-9000}
if [ ! -f scripts/serve_policy.py ]; then echo "Run this from the openpi folder (scripts/serve_policy.py not found)."; exit 1; fi
if [ -z "$task" ]; then echo "Usage: serve.sh <pnp|bit|wtp|base> [step] [experiment]"; exit 1; fi
if [ "$task" = base ]; then
  config=pi05_droid; dir=gs://openpi-assets/checkpoints/pi05_droid; name=base
else
  if [ -z "$step" ] || [ ! -d "checkpoints/$task/$exp/$step" ]; then
    echo "Saved steps in checkpoints/$task/$exp:"; ls "checkpoints/$task/$exp" 2>/dev/null || echo "  (none: is the training finished?)"; exit 1
  fi
  config=$task; dir=checkpoints/$task/$exp/$step; name=$exp/$step
fi
export PATH=$HOME/.local/bin:$PATH
export OPENPI_DATA_HOME="$PWD/openpi_cache"
echo "============================================================"
echo "Serving $name on $(hostname):$port. On the robot PC:"
echo "  ssh -N -J $USER@tinkercliffs1.arc.vt.edu -L $port:localhost:$port $USER@$(hostname)"
echo "  python3 -m demo.rollout ${task/base/<task>} --policy $name"
echo "============================================================"
uv run scripts/serve_policy.py --port "$port" policy:checkpoint --policy.config="$config" --policy.dir="$dir"
