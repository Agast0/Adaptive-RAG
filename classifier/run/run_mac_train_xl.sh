#!/usr/bin/env bash
# Mac-compatible classifier train/valid/predict cycle.
# Differences from run_large_train_xl.sh:
#   - drops CUDA_VISIBLE_DEVICES (no NVIDIA GPU on Mac)
#   - uses smaller t5-small base model and reduced batch/epoch counts
#   - single epoch sweep (1) instead of [15,20,25,30,35]
set -e
set -u
set -o pipefail 2>/dev/null || true

DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=t5-small
LLM_NAME=flan_t5_xl
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
EPOCH=1
BATCH=8

TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/epoch/${EPOCH}/${DATE}
mkdir -p ${TRAIN_OUTPUT_DIR}

python run_classifier.py \
    --model_name_or_path ${MODEL} \
    --train_file ./data/musique_hotpot_wiki2_nq_tqa_sqd/${LLM_NAME}/binary_silver/train.json \
    --question_column question \
    --answer_column answer \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_train_batch_size ${BATCH} \
    --output_dir ${TRAIN_OUTPUT_DIR} \
    --overwrite_cache \
    --train_column 'train' \
    --do_train \
    --num_train_epochs ${EPOCH}

VALID_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/valid
mkdir -p ${VALID_OUTPUT_DIR}
python run_classifier.py \
    --model_name_or_path ${TRAIN_OUTPUT_DIR} \
    --validation_file ./data/musique_hotpot_wiki2_nq_tqa_sqd/${LLM_NAME}/silver/valid.json \
    --question_column question \
    --answer_column answer \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 32 \
    --output_dir ${VALID_OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval

PREDICT_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/predict
mkdir -p ${PREDICT_OUTPUT_DIR}
python run_classifier.py \
    --model_name_or_path ${TRAIN_OUTPUT_DIR} \
    --validation_file ./data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
    --question_column question \
    --answer_column answer \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 32 \
    --output_dir ${PREDICT_OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval

echo "Train output dir: ${TRAIN_OUTPUT_DIR}"
