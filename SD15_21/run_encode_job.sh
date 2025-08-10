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
