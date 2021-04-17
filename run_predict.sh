MAX_LENGTH=128
BERT_MODEL=/home/whou/workspace/pretrained_models/bert-base-cased
OUTPUT_DIR=./output/ner
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
NUM_EPOCHS=45
WARMUP_STEPS=300
SAVE_STEPS=300
SEED=1
LR=3e-5

mkdir -p $OUTPUT_DIR 
# CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_ner_multi_task.py \
CUDA_VISIBLE_DEVICES=0  python3 run_ner.py  \
--data_dir ./data/conll-2003 \
--model_type bert \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict \
--overwrite_cache \
--overwrite_output_dir  > $OUTPUT_DIR/predict.log 2>&1 &
