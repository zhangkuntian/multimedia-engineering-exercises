#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"
DDIR=../data/
OUTDIR=output_dir_01

TRAIN=train_questions.json
DEV=dev1_questions.json
# TEST=dev2_questions.json
TEST=aio_leaderboard.json
ENTITY=candidate_entities.json.gz

MODEL="ALINEAR/albert-japanese-v2"
MODEL_TYPE="albert"
# MODEL="bandainamco-mirai/distilbert-base-japanese"
# MODEL_TYPE="distilbert"
# MODEL="cl-tohoku/bert-base-japanese-whole-word-masking"
# MODEL_TYPE="bert"

python3 save_predicted_logits.py \
  --data_dir ${DDIR} \
  --model_name_or_path ${MODEL} \
  --model_type ${MODEL_TYPE} \
  --task_name jaqket \
  --entities_fname ${ENTITY} \
  --dev_fname ${DEV} \
  --test_fname ${TEST} \
  --output_dir ${OUTDIR} \
  --do_eval \
  --do_test \
  --learning_rate=3e-5 \
  --num_train_epochs 10 \
  --logging_steps 100
