#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"
DDIR=../data/
OUTDIR=output_dir_00

TRAIN=train_questions.json
DEV=dev1_questions.json
TEST=dev2_questions.json
ENTITY=candidate_entities.json.gz

python3 main.py  \
  --data_dir   ${DDIR} \
  --model_name_or_path ALINEAR/albert-japanese-v2 \
  --task_name jaqket \
  --entities_fname ${ENTITY} \
  --train_fname ${TRAIN} \
  --dev_fname   ${DEV} \
  --test_fname  ${TEST} \
  --output_dir ${OUTDIR} \
  --num_options 2 \
  --do_train \
  --do_eval \
  --do_test \
  --per_gpu_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --logging_steps 10
