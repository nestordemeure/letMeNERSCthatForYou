#!/bin/bash

# load dependencies
module load python/3.10 cudatoolkit/12.0 cudnn/8.9.3_cuda12 pytorch/2.1.0-cu12
conda activate /global/cfs/cdirs/nstaff/chatbot/conda/chatbot

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# main folders
code_folder="$chatbot_root/production_code"
dev_code_folder="$chatbot_root/letMeNERSCthatForYou"
models_folder="$chatbot_root/models"
data_folder="$code_folder/data"
# data folders
documentation_folder="$data_folder/nersc_doc/docs"
database_folder="$data_folder/database"
python_instance="$chatbot_root/conda/chatbot/bin/python3"

# runs the worker
# Using python_instance to run the chatbot script in code_folder
$python_instance $dev_code_folder/local_async_stresstester.py --docs_folder $documentation_folder --database_folder $database_folder --models_folder $models_folder
