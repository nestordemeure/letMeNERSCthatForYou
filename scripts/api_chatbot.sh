#!/bin/bash

# load dependencies
module load python/3.10
conda activate /global/cfs/cdirs/nstaff/chatbot/conda/chatbot

# folders
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
code_folder="$chatbot_root/production_code"
python_instance="$chatbot_root/conda/chatbot/bin/python3"

# runs the client
# Using python_instance to run the api_client script in code_folder
$python_instance $code_folder/api_client.py
