#!/bin/bash

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# main folders
code_folder="$chatbot_root/production_code"
models_folder="$chatbot_root/models"
data_folder="$code_folder/data"
# data folders
documentation_folder="$data_folder/nersc_doc/docs"
database_folder="$data_folder/database"
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

# Change the current directory to the code folder
cd $code_folder

# Perform a git pull from origin main
git pull origin main

# runs the worker
# Using python_instance to run the api_consumer script in code_folder
$python_instance $code_folder/api_async_consumer.py --docs_folder $documentation_folder --database_folder $database_folder --models_folder $models_folder
