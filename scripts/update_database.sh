#!/bin/bash

# load dependencies
module load python cudatoolkit cudnn
conda activate /global/cfs/cdirs/nstaff/chatbot/conda

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# main folders
code_folder="$chatbot_root/letMeNERSCthatForYou"
models_folder="$chatbot_root/models"
data_folder="$code_folder/data"
# data folders
documentation_folder="$data_folder/nersc_doc/docs"
database_folder="$data_folder/database"
python_instance="$chatbot_root/conda/bin/python3"

# Change the current directory to the documentation folder
cd $documentation_folder

# Perform a git pull from origin main
git pull origin main

# Update the documentation
# Using python_instance to run the update_database script in code_folder
$python_instance $code_folder/update_database.py --docs_folder $documentation_folder --database_folder $database_folder --models_folder $models_folder

