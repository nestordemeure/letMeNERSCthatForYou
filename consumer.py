import json
import time
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
    parser.add_argument("--min_refresh_time", default=5, type=float, help="How many seconds should we wait before calls to the API")
    parser.add_argument("--api_key", default=None, help="the API key used to access NERSC services")
    parser.add_argument("--verbose", default=True, action='store_true', help="should we display messages as we run for debug purposes")
    args = parser.parse_args()
    return args

def main():
    # process command line arguments
    args= parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    models_folder = args.models_folder
    min_refresh_time = args.min_refresh_time
    api_key = args.api_key
    verbose = args.verbose

    # API details
    input_endpoint = "https://api-dev.nersc.gov/api/v1.2/ai/docs/work"
    output_endpoint = "https://api-dev.nersc.gov/api/v1.2/ai/docs/work_results"
    # TODO use the api key for security reasons

    # initializes models
    llm = lmntfy.models.llm.Vicuna(models_folder)
    embedder = lmntfy.models.embedding.SBERTEmbedding(models_folder)
    database = lmntfy.database.FaissDatabase(llm, embedder, docs_folder, database_folder, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, embedder, database)

    # run a loop to check on files
    # TODO harden it against errors
    if verbose: lmntfy.user_interface.command_line.display_logo()
    while True:
        # gets conversations as a json
        get_time = time.time()
        conversations = requests.get(input_endpoint).json()
        if verbose: print(f"\nGET:\n{json.dumps(conversations, indent=4)}")
        # loop on all conversation
        for id, messages in conversations.items():
            # gets an answer from the model
            try:
                answer = question_answerer.continue_chat(messages, verbose=False)
            except Exception as e:
                answer = {'role':'assistant', 'content': f"ERROR: {str(e)}"}
            # post the answer with the conversation key
            output={id: [answer]}
            requests.post(output_endpoint, data=output)
            if verbose: print(f"POST:\n{json.dumps(output, indent=4)}")
        # calculate how long the answering took
        # if it took less than min_refresh_time seconds, sleep for the remaining time
        elapsed_time = time.time() - get_time
        if elapsed_time < min_refresh_time:
            time.sleep(min_refresh_time - elapsed_time)
 
if __name__ == "__main__":
    main()
