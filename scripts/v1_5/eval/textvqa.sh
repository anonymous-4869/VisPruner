#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.5-7b"
SPLIT="llava_textvqa_val_v051_ocr"

METHOD="vispruner"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/textvqa/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl
