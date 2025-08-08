#!/bin/bash

# Create required directories
mkdir -p metrics keys results logs

# Parameters
ATTACKS=("stealthy" "white_noise"  )
EPS_VALUES=(4.0 6.0 8.0 10.0 12.0)
INVERSIONS=("null" "exact" )

# Fixed defaults
TEST_NUM=100
INF_STEPS=50
METHOD="prc"
MODEL_ID="stabilityai/stable-diffusion-2-1-base"
DATASET_ID="Gustavosta/Stable-Diffusion-Prompts"
FPR=0.1
NOWM=0
PRC_T=3
MESSAGE_LENGTH=1750
python3 make_prc_imgs.py \
  --test_num $TEST_NUM \
  --method prc \
  --model_id $MODEL_ID \
  --device "cuda:1" \
  --message_length $MESSAGE_LENGTH

if [ $? -ne 0 ]; then
  echo "❌ Failed: make_prc_imgs.py"
  exit 1
fi
# Devices
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3")
NUM_DEVICES=${#DEVICES[@]}

# Job counter for round-robin GPU assignment
device_idx=0

# Temporary script to run each job
JOB_SCRIPT="run_single_job.sh"

cat > $JOB_SCRIPT << 'EOF'
#!/bin/bash
# This script runs a single experiment
python3 encode_exact_invert_remove.py \
    --test_num $TEST_NUM \
    --method $METHOD \
    --model_id "$MODEL_ID" \
    --dataset_id "$DATASET_ID" \
    --inf_steps $INF_STEPS \
    --fpr $FPR \
    --eps $EPS \
    --attack "$ATTACK" \
    --nowm $NOWM \
    --prc_t $PRC_T \
    --message_length $MESSAGE_LENGTH \
    --inversion "$INVERSION" \
    --device "$DEVICE" \
    > "logs/$EXP_ID.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Completed: $EXP_ID"
else
    echo "❌ Failed: $EXP_ID"
fi
EOF

chmod +x $JOB_SCRIPT

# Function to launch a job
launch_job() {
    local attack=$1
    local eps=$2
    local inversion=$3
    local device=$4
    local exp_id=$5

    # Export variables for subshell
    export TEST_NUM INF_STEPS METHOD MODEL_ID DATASET_ID FPR NOWM PRC_T MESSAGE_LENGTH
    export ATTACK="$attack" EPS="$eps" INVERSION="$inversion" DEVICE="$device" EXP_ID="$exp_id"

    # Run in background
    ./"$JOB_SCRIPT" &
}

# Loop through all combinations
for attack in "${ATTACKS[@]}"; do
  for eps in "${EPS_VALUES[@]}"; do
    for inversion in "${INVERSIONS[@]}"; do
      # Assign device in round-robin fashion
      device=${DEVICES[$device_idx]}
      device_idx=$(( (device_idx + 1) % NUM_DEVICES ))

      # Construct exp_id
      exp_id="${METHOD}_num_${TEST_NUM}_steps_${INF_STEPS}_fpr_${FPR}_nowm_${NOWM}_eps_${eps}_attack_${attack}_mess_len_${MESSAGE_LENGTH}_inversion_${inversion}_model_${MODEL_ID//\//_}"

      # Launch job and limit concurrency
      launch_job "$attack" "$eps" "$inversion" "$device" "$exp_id"

      # Limit to number of GPUs
      if (( $(jobs -r | wc -l) >= NUM_DEVICES )); then
          wait -n  # Wait for any one job to finish before launching next
      fi
    done
  done
done

# Wait for all jobs to complete
wait
echo "All experiments completed!"

# Optional: Clean up
rm -f $JOB_SCRIPT