#!/bin/bash
# Shell script for PRC experiments
# Uses set -euo pipefail for safety
set -euo pipefail

# === Create required directories ===
mkdir -p metrics keys results logs

# === Parameters ===
MESSAGE_LENGTHS=(1600)                              # bits of message
DEVICES=("cuda:1" "cuda:2" "cuda:3")                         # adjust to your system
BOUNDARY_HIDING=0                                   # 1 enabled, 0 disabled

# Attack parameters
ATTACKS=("stealthy" "white_noise")
EPS_VALUES=(4 6 8 10)
INVERSIONS=("null" "prompt")                                 # AC1: exact, AC2: prompt, AC3: null

# Fixed defaults
TEST_NUM=50
INF_STEPS=50
METHOD="prc"
MODEL_ID="stabilityai/stable-diffusion-2-1-base"
DATASET_ID="Gustavosta/Stable-Diffusion-Prompts"
FPR=0.1
NOWM=0
PRC_T=3

NUM_DEVICES=${#DEVICES[@]}

# Export common vars for child scripts
export TEST_NUM METHOD MODEL_ID DATASET_ID INF_STEPS FPR NOWM PRC_T BOUNDARY_HIDING



# === PHASE 2: Encode / Invert / Remove ===
echo "🔹 Phase 2: Running encode_exact_invert_remove.py..."

ENC_JOB_SCRIPT="run_encode_job.sh"
cat > "$ENC_JOB_SCRIPT" << EOF
#!/bin/bash
python3 encode_exact_invert_remove_old.py \
  --test_num "\$TEST_NUM" \
  --method "\$METHOD" \
  --model_id "\$MODEL_ID" \
  --dataset_id "\$DATASET_ID" \
  --inf_steps "\$INF_STEPS" \
  --fpr "\$FPR" \
  --nowm "\$NOWM" \
  --prc_t "\$PRC_T" \
  --eps "\$EPS" \
  --attack "\$ATTACK" \
  --inversion "\$INVERSION" \
  --device "\$DEVICE" \
  --boundary_hiding "\$BOUNDARY_HIDING" \
  --message_length "\$MESSAGE_LENGTH" \
  > "logs/\$EXP_ID.log" 2>&1

if [ \$? -eq 0 ]; then
  echo "✅ Completed: \$EXP_ID"
else
  echo "❌ Failed: \$EXP_ID"
fi
EOF
chmod +x "$ENC_JOB_SCRIPT"

launch_encode_job() {
  local attack="$1" eps="$2" inversion="$3" device="$4" exp_id="$5" message_length="$6"
  export ATTACK="$attack" EPS="$eps" INVERSION="$inversion" DEVICE="$device" EXP_ID="$exp_id" MESSAGE_LENGTH="$message_length"
  "./$ENC_JOB_SCRIPT" &
}

device_idx=0
for message_length in "${MESSAGE_LENGTHS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    for eps in "${EPS_VALUES[@]}"; do
      for inversion in "${INVERSIONS[@]}"; do
        device="${DEVICES[$device_idx]}"
        device_idx=$(( (device_idx + 1) % NUM_DEVICES ))

        exp_id="prc_num_${TEST_NUM}_steps_${INF_STEPS}_fpr_${FPR}_nowm_${NOWM}_eps_${eps}_attack_${attack}_mess_len_${message_length}_inversion_${inversion}_model_stabilityai_stable-diffusion-2-1-base"

        launch_encode_job "$attack" "$eps" "$inversion" "$device" "$exp_id" "$message_length"

        if (( $(jobs -r | wc -l) >= NUM_DEVICES )); then
          wait -n || true
        fi
      done
    done
  done
done

echo "⏳ Waiting for all encode jobs..."
wait

# === Done ===
rm -f  "$ENC_JOB_SCRIPT"
echo "🎉 All experiments completed successfully!"
echo "📄 Logs saved in ./logs/"
