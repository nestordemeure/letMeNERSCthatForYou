#!/bin/bash

# Define the base chatbot folder
chatbot_root="/global/u2/n/nestor/scratch_perlmutter/chatbot"

# Define the variables
documentation_folder="$chatbot_root/documentation/docs"
database_folder="$chatbot_root/database"
models_folder="$chatbot_root/models"
chatbot_folder="$chatbot_root/code"
python_instance="$chatbot_root/python/chatbot_env/bin/python3"

# Change the current directory to the documentation folder
cd $documentation_folder

# Perform a git pull from origin main
git pull origin main

# Update the documentation
# Using python_instance to run the update_database script in chatbot_folder
$python_instance $chatbot_folder/update_database.py --docs_folder $documentation_folder --database_folder $database_folder --models_folder $models_folder

