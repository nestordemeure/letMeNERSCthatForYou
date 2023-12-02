#!/bin/bash

# Define the base chatbot folder
chatbot_root="/global/u2/n/nestor/scratch_perlmutter/chatbot"

# Define the variables
documentation_folder="$chatbot_root/documentation/docs"
database_folder="$chatbot_root/database"
models_folder="$chatbot_root/models"
chatbot_folder="$chatbot_root/code"
python_instance="$chatbot_root/python/chatbot_env/bin/python3"

# runs the worker
# Using python_instance to run the api_consumer script in chatbot_folder
$python_instance $chatbot_folder/api_consumer.py --docs_folder $documentation_folder --database_folder $database_folder --models_folder $models_folder

