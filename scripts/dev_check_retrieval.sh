#!/bin/bash

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# local folders
script_dir=$(dirname "$(realpath "$0")")
dev_code_folder=$(dirname "$script_dir")
# main folders
code_folder="$chatbot_root/production_code"
models_folder="$chatbot_root/models"
data_folder="$code_folder/data"
# data folders
documentation_folder="$data_folder/nersc_doc/docs"
database_folder="$data_folder/database"
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

# Update the documentation
# Using python_instance to run the update_database script in code_folder
$python_instance $dev_code_folder/check_retrieval.py --docs_folder $documentation_folder --database_folder $database_folder --models_folder $models_folder

