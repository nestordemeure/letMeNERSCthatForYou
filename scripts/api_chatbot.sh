#!/bin/bash

# load dependencies
module load python
conda activate /global/cfs/cdirs/nstaff/chatbot/conda

# folders
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
code_folder="$chatbot_root/letMeNERSCthatForYou"
python_instance="$chatbot_root/conda/bin/python3"

# runs the client
# Using python_instance to run the api_client script in code_folder
$python_instance $code_folder/api_client.py
