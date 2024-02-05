import json
import time
import asyncio
import aiohttp
import lmntfy
import argparse
from pathlib import Path
from lmntfy.user_interface.web import SFAPIOAuthClient

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder")
    parser.add_argument("--models_folder", default="../models", type=Path, help="path to the folder containing all the models")
    parser.add_argument("--min_refresh_time", default=1, type=float, help="minimum seconds to wait before calling the API again")
    parser.add_argument("--max_refresh_time", default=5, type=float, help="maximum seconds to wait before calling the API again")
    parser.add_argument("--cooldown_time", default=300, type=float, help="time in seconds to switch from min to max refresh time due to inactivity")
    parser.add_argument("--max_concurrent_tasks", default=100, type=int, help="maximum number of concurrent tasks")
    parser.add_argument("--api_key", default=None, help="the API key used to access NERSC services")
    parser.add_argument("--verbose", default=True, action='store_true', help="enable verbose output for debugging purposes")
    return parser.parse_args()

async def process_conversation(semaphore, session, oauth_client, output_endpoint, question_answerer, id, messages, verbose):
    """
    Process an individual conversation by generating a response and posting it to the output endpoint.

    This function first attempts to generate a response using the `question_answerer` model. 
    If successful, the answer is sent back to the specified output endpoint. 
    If an exception occurs during answer generation, an error message is constructed and sent instead.

    Args:
    - semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent executions.
    - session (aiohttp.ClientSession): The session used for making HTTP requests.
    - oauth_client (SFAPIOAuthClient): The SFAPIOAuthClient to connect to the API
    - output_endpoint (str): The URL to which the generated answer should be posted.
    - question_answerer (QuestionAnswerer): The model used for generating answers to the incoming messages.
    - id (str or int): The identifier of the conversation to which the messages belong.
    - messages (list): The list of messages for which the answer is to be generated.
    - verbose (bool): If True, additional details (such as POST request status) will be printed to the console.

    This function ensures that the number of concurrent executions does not exceed the limit set by the semaphore.
    """
    async with semaphore:
        # Generates an answer using the question_answerer model.
        try:
            answer = question_answerer.chat(messages, verbose=False)
        except Exception as e:
            # Constructs an error message in case of an exception during answer generation.
            answer = {'role': 'assistant', 'content': f"ERROR: {str(e)}"}
        output = {id: [answer]}

        # Sends the generated answer or error message to the output endpoint.
        headers = {'accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': oauth_client.get_authorization_header()['Authorization']}
        async with session.post(output_endpoint, json=output, headers=headers) as response:
            status = response.status  # Retrieves the status code of the POST request.

        # Optionally prints details of the POST request if verbose mode is enabled.
        if verbose:
            print(f"POST (status code:{status}):\n{json.dumps(output, indent=4)}")

async def wait_for_next_iteration(last_active_time, start_time, min_refresh_time, max_refresh_time, cooldown_time):
    """
    Determine the appropriate wait time before the next iteration based on user activity.
    
    If a user message was received within the cooldown_time, the function waits for min_refresh_time.
    Otherwise, it waits for max_refresh_time.
    
    Args:
    - last_active_time: The timestamp of the last received user message.
    - start_time: The timestamp when the current iteration started.
    - min_refresh_time: The minimum time to wait before the next API call if recently active.
    - max_refresh_time: The maximum time to wait before the next API call if not recently active.
    - cooldown_time: The time window to consider for recent activity.
    """
    # Determine the refresh time based on user activity
    current_time = time.time()
    if current_time - last_active_time < cooldown_time:
        refresh_time = min_refresh_time
    else:
        refresh_time = max_refresh_time

    # Calculate how long the answering took
    elapsed_time = current_time - start_time

    # Sleep for the remaining time if the processing was faster than refresh_time
    if elapsed_time < refresh_time:
        await asyncio.sleep(refresh_time - elapsed_time)

async def main():
    # Initialize models and API details
    args = parse_args()
    llm = lmntfy.models.llm.Zephyr(args.models_folder)
    embedder = lmntfy.models.embedding.SBERTEmbedding(args.models_folder)
    database = lmntfy.database.FaissDatabase(llm, embedder, args.docs_folder, args.database_folder, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, embedder, database)
    semaphore = asyncio.Semaphore(args.max_concurrent_tasks)

    # API details
    api_base_url = "https://api-dev.nersc.gov/api/v1.2"
    input_endpoint = f"{api_base_url}/ai/docs/work"
    output_endpoint = f"{api_base_url}/ai/docs/work_results"
    oauth_client = SFAPIOAuthClient(api_base_url=api_base_url)

    async with aiohttp.ClientSession() as session:
        if args.verbose: 
            lmntfy.user_interface.command_line.display_logo()

        running_tasks = []
        last_active_time = time.time()  # Track the last time a message was received
        while True:
            start_time = time.time()

            # Filter out the completed tasks
            running_tasks = [task for task in running_tasks if not task.done()]

            # Get conversations as JSON
            async with session.get(input_endpoint, headers=oauth_client.get_authorization_header()) as response:
                try:
                    # Parses the answer as a json
                    conversations = await response.json()
                except aiohttp.client_exceptions.ContentTypeError as e:
                    # Parsing failed
                    # Get the raw response text
                    response_text = await response.text()
                    # Displays (for logs) an error message with response details
                    print(
                        f"ContentTypeError when trying to parse JSON from the response.\n"
                        f"Status: {response.status}, Content-Type: {response.headers.get('Content-Type')}\n"
                        f"Response body:\n{response_text}")
                    # Wait for max_refresh_time before skipping to the next iteration
                    await asyncio.sleep(args.max_refresh_time)
                    continue

            if args.verbose: 
                print(f"\nGET:\n{json.dumps(conversations, indent=4)}")

            # Update last_active_time when a message is received
            if len(conversations) > 0:
                last_active_time = time.time()

            # Process the messages
            for id, messages in conversations.items():
                await semaphore.acquire() # Wait for an available slot in the semaphore before creating a new task
                task = asyncio.create_task(process_conversation(semaphore, session, oauth_client, output_endpoint, question_answerer, id, messages, args.verbose))
                task.add_done_callback(lambda t: semaphore.release())  # Release semaphore when task is done
                running_tasks.append(task)

            # Wait until the next api call
            await wait_for_next_iteration(last_active_time, start_time, args.min_refresh_time, args.max_refresh_time, args.cooldown_time)

if __name__ == "__main__":
    asyncio.run(main())
