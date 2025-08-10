#!/bin/bash

# Create required directories
mkdir -p metrics keys results logs

# === Parameters ===
MESSAGE_LENGTHS=(1000 1100 1200 1300 1400 1500)
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3")
NUM_DEVICES=${#DEVICES[@]}

# Fixed defaults for make_prc_imgs.py
TEST_NUM=100
INF_STEPS=50
METHOD="prc"
MODEL_ID="stabilityai/stable-diffusion-2-1-base"
DATASET_ID="Gustavosta/Stable-Diffusion-Prompts"
FPR=0.1
NOWM=0
PRC_T=3

# Job script for parallel execution
JOB_SCRIPT="run_make_prc.sh"
cat > "$JOB_SCRIPT" << 'EOF'
#!/bin/bash
# Run make_prc_imgs.py with given message_length and device
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
  --message_length "$MESSAGE_LENGTH" > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Completed: message_length=$MESSAGE_LENGTH on $DEVICE"
else
  echo "❌ Failed: message_length=$MESSAGE_LENGTH" | tee -a failed.log
fi
EOF

chmod +x "$JOB_SCRIPT"

# === PHASE 1: Generate images in parallel ===
echo "🔹 Phase 1: Running make_prc_imgs.py in parallel..."

device_idx=0
for message_length in "${MESSAGE_LENGTHS[@]}"; do
  device=${DEVICES[$device_idx]}
  device_idx=$(( (device_idx + 1) % NUM_DEVICES ))

  # Construct exp_id (must match your Python script logic)
  exp_id="prc_num_${TEST_NUM}_steps_${INF_STEPS}_fpr_${FPR}_nowm_${NOWM}_eps_10.0_attack_stealthy_mess_len_${message_length}_inversion_null_model_stabilityai_stable-diffusion-2-1-base"

  # Log file
  LOG_FILE="logs/${exp_id}_make_prc.log"

  # Export variables and run in background
  export TEST_NUM METHOD MODEL_ID DATASET_ID INF_STEPS FPR NOWM PRC_T
  export MESSAGE_LENGTH="$message_length" DEVICE="$device" LOG_FILE="$LOG_FILE"

  ./"$JOB_SCRIPT" &

  # Limit concurrency: wait if we've reached max GPU count
  if (( $(jobs -r | wc -l) >= NUM_DEVICES )); then
    wait -n  # Wait for any job to finish before launching next
  fi
done

# ✅ CRITICAL: Wait for ALL make_prc_imgs jobs to complete
echo "⏳ Waiting for all make_prc_imgs.py jobs to finish..."
wait
echo "✅ Phase 1 completed: All make_prc_imgs.py jobs have terminated."


# === PHASE 2: Run encode_exact_invert_remove.py experiments ===
echo "🔹 Phase 2: Running encode_exact_invert_remove.py across attack/eps/inversion..."

# Parameters for attack/inversion
ATTACKS=("white_noise" "stealthy")
EPS_VALUES=(4 6 8 10 12)
INVERSIONS=("null" "exact")



# Temporary script for encode jobs
ENC_JOB_SCRIPT="run_encode_job.sh"
cat > "$ENC_JOB_SCRIPT" << 'EOF'
#!/bin/bash
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
  --message_length "$MESSAGE_LENGTH" \
  > "logs/$EXP_ID.log" 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Completed: $EXP_ID"
else
  echo "❌ Failed: $EXP_ID"
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

  export ENC_TEST_NUM METHOD MODEL_ID DATASET_ID INF_STEPS FPR NOWM PRC_T
  export ATTACK="$attack" EPS="$eps" INVERSION="$inversion" DEVICE="$device"
  export EXP_ID="$exp_id" MESSAGE_LENGTH="$message_length"

  ./"$ENC_JOB_SCRIPT" &
}

# Loop over all message lengths and parameters
device_idx=0
for message_length in "${MESSAGE_LENGTHS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    for eps in "${EPS_VALUES[@]}"; do
      for inversion in "${INVERSIONS[@]}"; do
        device=${DEVICES[$device_idx]}
        device_idx=$(( (device_idx + 1) % NUM_DEVICES ))

        # Construct exp_id (must match your script)
        exp_id="prc_num_${ENC_TEST_NUM}_steps_${INF_STEPS}_fpr_${FPR}_nowm_${NOWM}_eps_${eps}_attack_${attack}_mess_len_${message_length}_inversion_${inversion}_model_stabilityai_stable-diffusion-2-1-base"

        launch_encode_job "$attack" "$eps" "$inversion" "$device" "$exp_id"

        if (( $(jobs -r | wc -l) >= NUM_DEVICES )); then
          wait -n
        fi
      done
    done
  done
done

# Final wait
wait
echo "🎉 All experiments completed!"
rm -f "$JOB_SCRIPT" "$ENC_JOB_SCRIPT"