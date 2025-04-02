#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.5-7b"
SPLIT="mmbench_dev_cn_20231003"

METHOD="vispruner"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATA_DIR}/mmbench_cn/${SPLIT}.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench_cn/answers_upload/${SPLIT}/${CKPT}/${METHOD}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench_cn/${SPLIT}.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/${SPLIT}/${CKPT}/${METHOD} \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/${SPLIT}/${CKPT}/${METHOD} \
    --experiment ${PARAM}
