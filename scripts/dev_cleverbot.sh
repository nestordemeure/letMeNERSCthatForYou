#!/bin/bash

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# local folders
script_dir=$(dirname "$(realpath "$0")")
dev_code_folder=$(dirname "$script_dir")
# main folders
code_folder="$chatbot_root/production_code"
models_folder="$chatbot_root/models"
# data folders
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

# runs the worker
# Using python_instance to run the chatbot script in dev_code_folder
$python_instance $dev_code_folder/cleverbot.py --models_folder $models_folder
