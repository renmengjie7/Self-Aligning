MODEL_PATH=${1}
MODEL_NAME=${2}

vicuna=${3}
shot=${4}
split=${5}

lang=${6}
task=${7}

DATA_PATH=${8}
OUTPUT_PATH=${9}

eval_type=${10}
num_gpus=${11}
hint=${12}

ARGS="
--model-path ${MODEL_PATH} \
--model-name ${MODEL_NAME} \
--vicuna ${vicuna} \
--shot ${shot} \
--split ${split} \
--lang-prompt ${lang} \
--task ${task} \
--data-dir ${DATA_PATH} \
--output-dir ${OUTPUT_PATH} \
--eval-type ${eval_type} \
--num-gpus ${num_gpus} \
--hint ${hint}
"

RUN="${HOME}/.conda/envs/eval/bin/python my_benchmark_eval.py"

echo $RUN $ARGS
$RUN $ARGS

