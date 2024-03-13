import json
import time
import requests
import lmntfy
import argparse
from pathlib import Path
from lmntfy.user_interface.web import SFAPIOAuthClient

# use the dev side of the API
API_BASE_URL='https://api-dev.nersc.gov/api/internal/v1.2'
TOKEN_URL='https://oidc-dev.nersc.gov/c2id/token'

def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder", default="../models",type=Path, help="path to the folder containing all the models")
    parser.add_argument("--min_refresh_time", default=5, type=float, help="How many seconds should we wait before calls to the API")
    parser.add_argument("--api_key", default=None, help="the API key used to access NERSC services")
    parser.add_argument("--verbose", default=True, action='store_true', help="should we display messages as we run for debug purposes")
    args = parser.parse_args()
    return args

def get_conversation(input_endpoint:str, oauth_client:SFAPIOAuthClient, verbose=False):
    """
    GET the conversations stored in the API
    """
    conversations = requests.get(input_endpoint, headers=oauth_client.get_authorization_header()).json()
    if verbose: print(f"\nGET:\n{json.dumps(conversations, indent=4)}")
    return conversations

def post_answer(output_endpoint:str, oauth_client:SFAPIOAuthClient, conversation_key:str, answer, verbose=False):
    """
    POSTS an answer to the API
    """
    output={conversation_key: [answer]}
    headers = {'accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': oauth_client.get_authorization_header()['Authorization']}
    response = requests.post(output_endpoint, json=output, headers=headers)
    if verbose: print(f"POST (status code:{response.status_code}):\n{json.dumps(answer, indent=4)}")
    return response

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
    input_endpoint = f"{API_BASE_URL}/ai/docs/work"
    output_endpoint = f"{API_BASE_URL}/ai/docs/work_results"
    oauth_client = SFAPIOAuthClient(api_base_url=API_BASE_URL, token_url=TOKEN_URL)

    # initializes models
    llm = lmntfy.models.llm.Default(models_folder, device='cuda')
    embedder = lmntfy.models.embedding.Default(models_folder, device='cuda')
    reranker = lmntfy.models.reranker.Default(models_folder, device='cuda')
    database = lmntfy.database.Default(docs_folder, database_folder, llm, embedder, reranker, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, embedder, database)

    # run a loop to check on files
    if verbose: lmntfy.user_interface.command_line.display_logo()
    while True:
        # gets conversations as a json
        get_time = time.time()
        conversations = get_conversation(input_endpoint, oauth_client, verbose=verbose)
        # loop on all conversation
        for id, messages in conversations.items():
            try:
                # gets an answer from the model
                answer = question_answerer.chat(messages, verbose=False)
                # post the answer with the conversation key
                post_answer(output_endpoint, oauth_client, id, answer, verbose=verbose)
            except Exception as e:
                # produce a polite answer in case of crash
                answer = {'role':'assistant', 'content': "Error: I am terribly sorry, but the Documentation chatbot is currently experiencing technical difficulties. Please try again in ten minutes or more."}
                # post the answer with the conversation key
                post_answer(output_endpoint, oauth_client, id, answer)
                # then crash
                raise
        # calculate how long the answering took
        # if it took less than min_refresh_time seconds, sleep for the remaining time
        elapsed_time = time.time() - get_time
        if elapsed_time < min_refresh_time:
            time.sleep(min_refresh_time - elapsed_time)
 
if __name__ == "__main__":
    main()
