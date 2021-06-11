DATASET=ccks
TASK=trigger
DOMAIN=few
DATA_DIR=./data/FewFC-main/rearranged/$DOMAIN
LABEL=./data/FewFC-main/event_schema/$DOMAIN.json
MAX_LENGTH=256
# BERT_MODEL=/home/whou/workspace/pretrained_models/bert-base-cased
BERT_MODEL=/hy-nas/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/
# BERT_MODEL=./output/ccks/base/identification/checkpoint-best/
OUTPUT_DIR=./output/$DATASET/base-\>few/multi_task/
BATCH_SIZE=16
EVAL_BATCH_SIZE=64
NUM_EPOCHS=10000
WARMUP_STEPS=100
SAVE_STEPS=100
SEED=1
LR=2e-5

mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0 python3 run_ner.py  \
# CUDA_VISIBLE_DEVICES=0 nohup python3 -u run_ner_joint.py  \
CUDA_VISIBLE_DEVICES=0 nohup python3 -u run_ner_multi_task.py  \
--dataset $DATASET \
--task $TASK \
--data_dir $DATA_DIR \
--labels $LABEL \
--model_type bert \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps 1 \
--save_steps $SAVE_STEPS \
--logging_steps $SAVE_STEPS \
--seed $SEED \
--do_eval \
--evaluate_during_training \
--early_stop 3 \
--learning_rate $LR \
--weight_decay 0 \
--warmup_steps $WARMUP_STEPS \
--seed $SEED \
--overwrite_cache \
--overwrite_output_dir > $OUTPUT_DIR/eval.log 2>&1 &