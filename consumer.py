import json
import requests
import lmntfy
import argparse
from pathlib import Path

def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="/global/u2/n/nestor/scratch_perlmutter/chatbot/documentation/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="/global/u2/n/nestor/scratch_perlmutter/chatbot/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder",default="/global/u2/n/nestor/scratch_perlmutter/chatbot/models",type=Path, help="path to the folder containing all the models")
    args = parser.parse_args()
    # Convert the question list back to a string
    args.question = " ".join(args.question).strip()
    return args

def main():
    # process command line arguments
    args= parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    models_folder = args.models_folder

    # API details
    input_endpoint = "https://api-dev.nersc.gov/ai/doc/work"
    output_endpoint = "https://api-dev.nersc.gov/ai/doc/work_results"
    # TODO use an api key (passed as input) for security reasons

    # initializes models
    llm = lmntfy.models.llm.Vicuna(models_folder)
    embedder = lmntfy.models.embedding.SBERTEmbedding(models_folder)
    database = lmntfy.database.FaissDatabase(llm, embedder, docs_folder, database_folder, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, embedder, database)

    # run a loop to check on files
    # TODO harden it against errors
    while True:
        # gets conversations as a json
        conversations = requests.get(input_endpoint).json()
        # loop on all conversation
        for id, messages in conversations.items():
            # gets an answer form the model
            answer = question_answerer.continue_chat(messages, verbose=False)['content']
            # post the answer with the conversation key
            output={id: answer}
            requests.post(output_endpoint, data=output)
 
if __name__ == "__main__":
    main()
