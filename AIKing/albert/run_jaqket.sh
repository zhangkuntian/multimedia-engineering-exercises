#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"
DDIR=../data/
OUTDIR=output_dir_01

TRAIN=train_questions.json
DEV=dev1_questions.json
TEST=dev2_questions.json
ENTITY=candidate_entities.json.gz

MODEL="ALINEAR/albert-japanese-v2"
MODEL_TYPE="albert"
# MODEL="cl-tohoku/bert-base-japanese-whole-word-masking"
# MODEL_TYPE="bert"

python3 jaqket_albert.py  \
  --data_dir ${DDIR} \
  --model_name_or_path ${MODEL} \
  --model_type ${MODEL_TYPE} \
  --task_name jaqket \
  --entities_fname ${ENTITY} \
  --train_fname ${TRAIN} \
  --dev_fname ${DEV} \
  --test_fname ${TEST} \
  --output_dir ${OUTDIR} \
  --train_num_options 20 \
  --do_train \
  --do_eval \
  --do_test \
  --learning_rate=3e-5 \
  --per_gpu_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --logging_steps 10 \
  --save_steps 5000
