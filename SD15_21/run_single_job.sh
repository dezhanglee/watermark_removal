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
