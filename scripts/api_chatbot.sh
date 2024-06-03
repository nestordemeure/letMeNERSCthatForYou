#!/bin/bash

# folders
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
code_folder="$chatbot_root/production_code"
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

# runs the client
# Using python_instance to run the api_client script in code_folder
$python_instance $code_folder/api_client.py
