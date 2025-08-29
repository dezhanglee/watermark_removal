#!/bin/bash
python3 make_prc_imgs.py   --test_num "$TEST_NUM"   --method "$METHOD"   --model_id "$MODEL_ID"   --dataset_id "$DATASET_ID"   --inf_steps "$INF_STEPS"   --nowm "$NOWM"   --fpr "$FPR"   --prc_t "$PRC_T"   --device "$DEVICE"   --boundary_hiding "$BOUNDARY_HIDING"   --message_length "$MESSAGE_LENGTH"   > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Completed: message_length=$MESSAGE_LENGTH on $DEVICE"
else
  echo "❌ Failed: message_length=$MESSAGE_LENGTH on $DEVICE" | tee -a failed.log
fi
