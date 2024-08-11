#! /bin/bash
HOME=xxx

NNODES=`python parse_environment.py nnodes`
MASTER_ADDR=`python parse_environment.py master_addr`
MASTER_PORT=`python parse_environment.py master_port`
GPUS_PER_NODE=`python parse_environment.py nproc_per_node`
NODE_RANK=`python parse_environment.py node_rank`

model_name=${1}
model_path=${2}
data_name=${3}
data_path=${4}
save_subdir=${5}
micro_bs=${6}
acc_steps=${7}
n_gpus=${8}
bs=$((micro_bs * acc_steps * n_gpus))

model_name_or_path=${HOME}/models/mydownload/${model_path}
train_file=${HOME}/${data_path}
output_dir=${HOME}/models/trained/sft/domain/${save_subdir}/${model_name}-${data_name}-bs${bs}-ep3
eval_file=${HOME}/datas/processed/mmlu_dev_shot0_vicuna.jsonl

TRAIN_ARGS="
--num_train_epochs 3 \
--per_device_train_batch_size ${micro_bs} \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps ${acc_steps} \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--loss_input False \
"

EVAL_ARGS="
--evaluation_strategy "steps" \
--eval_steps 10000 \
"

DATA_ARGS="
--data_path ${train_file} \
--eval_data_path ${eval_file} \
"

MODEL_ARGS="
--model_name_or_path ${model_name_or_path}  \
--bf16 True \
--model_max_length 2048 \
--tf32 True \
--gradient_checkpointing True \
"

NO_SAVE_ARGS="
--save_strategy "steps" \
--save_steps 10000 \
--save_total_limit 100 \
--output_dir ${output_dir} \
--logging_dir ${output_dir}/tensorboard
"

SAVE_ARGS="
--save_strategy "epoch" \
--save_total_limit 100 \
--output_dir ${output_dir} \
--logging_dir ${output_dir}/tensorboard
"

ENV_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK}"

echo ${TRAIN_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${EVAL_ARGS} ${NO_SAVE_ARGS}

if [[ $model_name == *"mistral"* ]]; then
TRAIN_ARGS="
--num_train_epochs 3 \
--per_device_train_batch_size ${micro_bs} \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps ${acc_steps} \
--learning_rate 1e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--loss_input False \
"
${HOME}/.conda/envs/fschat_mistral/bin/torchrun ${ENV_ARGS} fastchat/train/train.py \
    ${TRAIN_ARGS} ${NO_SAVE_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${EVAL_ARGS} \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer'
elif [[ $model_name == *"70b"* ]]; then
${HOME}/.conda/envs/fschat_dpsp/bin/torchrun ${ENV_ARGS} fastchat/train/train.py \
    ${TRAIN_ARGS} ${SAVE_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${EVAL_ARGS} \
    --deepspeed deepspeed_s3.json
else
${HOME}/.conda/envs/fschat/bin/torchrun ${ENV_ARGS} fastchat/train/train.py \
    ${TRAIN_ARGS} ${NO_SAVE_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${EVAL_ARGS} \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"
fi
