#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, and pipeline failures

# Create required directories
mkdir -p metrics keys results logs

# === Parameters ===
MESSAGE_LENGTHS=(1600)
DEVICES=("cuda:0")
NUM_DEVICES=${#DEVICES[@]}

# Fixed defaults
TEST_NUM=10
INF_STEPS=50
METHOD="prc"
MODEL_ID="stabilityai/stable-diffusion-2-1-base"
DATASET_ID="Gustavosta/Stable-Diffusion-Prompts"
FPR=0.1
NOWM=0
PRC_T=3
BOUNDARY_HIDING=1  # ✅ Corrected spelling

# Export common variables for child scripts
export TEST_NUM METHOD MODEL_ID DATASET_ID INF_STEPS FPR NOWM PRC_T BOUNDARY_HIDING

# === PHASE 1: Generate images in parallel ===
echo "🔹 Phase 1: Running make_prc_imgs.py in parallel..."

JOB_SCRIPT="run_make_prc.sh"
cat > "$JOB_SCRIPT" << 'EOF'
#!/bin/bash
# This script runs make_prc_imgs.py with passed environment variables

# Debug (uncomment if needed): echo "DEBUG: DEVICE=$DEVICE, MESSAGE_LENGTH=$MESSAGE_LENGTH, BOUNDARY_HIDING=$BOUNDARY_HIDING" >&2

python3 make_prc_imgs.py \
  --test_num "$TEST_NUM" \
  --method "$METHOD" \
  --model_id "$MODEL_ID" \
  --dataset_id "$DATASET_ID" \
  --inf_steps "$INF_STEPS" \
  --nowm "$NOWM" \
  --fpr "$FPR" \
  --prc_t "$PRC_T" \
  --device "$DEVICE" \
  --boundary_hiding "$BOUNDARY_HIDING" \
  --message_length "$MESSAGE_LENGTH" > "$LOG_FILE" 2>&1

if [ \$? -eq 0 ]; then
  echo "✅ Completed: message_length=$MESSAGE_LENGTH on $DEVICE"
else
  echo "❌ Failed: message_length=$MESSAGE_LENGTH on $DEVICE" | tee -a failed.log
fi
EOF

chmod +x "$JOB_SCRIPT"

# Launch Phase 1 jobs
device_idx=0
for message_length in "${MESSAGE_LENGTHS[@]}"; do
  device=${DEVICES[$device_idx]}
  device_idx=$(( (device_idx + 1) % NUM_DEVICES ))

  # Construct exp_id (must match your Python script logic)
  exp_id="prc_num_${TEST_NUM}_steps_${INF_STEPS}_fpr_${FPR}_nowm_${NOWM}_eps_10.0_attack_stealthy_mess_len_${message_length}_inversion_null_model_stabilityai_stable-diffusion-2-1-base"
  LOG_FILE="logs/${exp_id}_make_prc.log"

  # Export per-job variables
  export MESSAGE_LENGTH="$message_length" DEVICE="$device" LOG_FILE="$LOG_FILE"

  ./"$JOB_SCRIPT" &

  # Limit concurrent jobs to number of devices
  if (( $(jobs -r | wc -l) >= NUM_DEVICES )); then
    wait -n
  fi
done

# Wait for all Phase 1 jobs to complete
echo "⏳ Waiting for all make_prc_imgs.py jobs to finish..."
wait
echo "✅ Phase 1 completed: All image generation jobs finished."

# === PHASE 2: Run encode_exact_invert_remove.py ===
echo "🔹 Phase 2: Running encode_exact_invert_remove.py across attack/eps/inversion..."

# Parameters
ATTACKS=("white_noise" "stealthy")
EPS_VALUES=(4 6 8 10 12)
INVERSIONS=("prompt")

# Temporary script for encode jobs
ENC_JOB_SCRIPT="run_encode_job.sh"
cat > "$ENC_JOB_SCRIPT" << 'EOF'
#!/bin/bash
# This script runs encode_exact_invert_remove.py

python3 encode_exact_invert_remove.py \
  --test_num "$TEST_NUM" \
  --method "$METHOD" \
  --model_id "$MODEL_ID" \
  --dataset_id "$DATASET_ID" \
  --inf_steps "$INF_STEPS" \
  --fpr "$FPR" \
  --nowm "$NOWM" \
  --prc_t "$PRC_T" \
  --eps "$EPS" \
  --attack "$ATTACK" \
  --inversion "$INVERSION" \
  --device "$DEVICE" \
  --boundary_hiding "$BOUNDARY_HIDING" \
  --message_length "$MESSAGE_LENGTH" \
  > "logs/$EXP_ID.log" 2>&1

if [ \$? -eq 0 ]; then
  echo "✅ Completed: $EXP_ID"
else
  echo "❌ Failed: $EXP_ID" | tee -a failed.log
fi
EOF

chmod +x "$ENC_JOB_SCRIPT"

# Function to launch encode job
launch_encode_job() {
  local attack=$1
  local eps=$2
  local inversion=$3
  local device=$4
  local exp_id=$5

  export ATTACK EPS INVERSION DEVICE EXP_ID MESSAGE_LENGTH
  ./"$ENC_JOB_SCRIPT" &
}

# Launch Phase 2 jobs
device_idx=0
for message_length in "${MESSAGE_LENGTHS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    for eps in "${EPS_VALUES[@]}"; do
      for inversion in "${INVERSIONS[@]}"; do
        device=${DEVICES[$device_idx]}
        device_idx=$(( (device_idx + 1) % NUM_DEVICES ))

        # Construct exp_id (must match your Python script naming logic)
        exp_id="prc_num_${TEST_NUM}_steps_${INF_STEPS}_fpr_${FPR}_nowm_${NOWM}_eps_${eps}_attack_${attack}_mess_len_${message_length}_inversion_${inversion}_model_stabilityai_stable-diffusion-2-1-base"

        launch_encode_job "$attack" "$eps" "$inversion" "$device" "$exp_id"

        # Limit concurrent jobs
        if (( $(jobs -r | wc -l) >= NUM_DEVICES )); then
          wait -n
        fi
      done
    done
  done
done

# Wait for all Phase 2 jobs
echo "⏳ Waiting for all encode_exact_invert_remove.py jobs to finish..."
wait
echo "🎉 All experiments completed successfully!"

# Cleanup temporary scripts
rm -f "$JOB_SCRIPT" "$ENC_JOB_SCRIPT"

# Optional: summary
echo "📄 Logs are saved in ./logs/"