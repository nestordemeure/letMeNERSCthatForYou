#!/global/cfs/cdirs/nstaff/chatbot/conda/bin/python3
"""
API stress test with multiple clients chatting simultaneously.

This script is designed to simulate multiple clients sending messages to an API endpoint concurrently.
It uses asynchronous programming to handle multiple chat sessions in parallel, each repeating a fixed question multiple times.
"""
from lmntfy.user_interface.web import SFAPIOAuthClient
import aiohttp
import asyncio

# use the dev side of the API
API_BASE_URL='https://api-dev.nersc.gov/api/internal/v1.2'
TOKEN_URL='https://oidc-dev.nersc.gov/c2id/token'

async def get_answer(session, oauth_client, convo_id, messages, refresh_time: int=1):
    url = f'{oauth_client.api_base_url}/ai/docs?convo_id={convo_id}'
    headers = {'accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': oauth_client.get_authorization_header()['Authorization']}
    
    # Ensure the payload matches the structure the API expects, likely just the messages list directly
    async with session.post(url, json=messages, headers=headers) as response_post:
        if response_post.status != 200:
            text = await response_post.text()
            print(f"ERROR (status:{response_post.status}): {text}")
            return {'role': 'assistant', 'content': text}

        # Poll the API for an answer and return it when received
        while True:
            async with session.get(url, headers=oauth_client.get_authorization_header()) as response_get:
                try:
                    # Parses the answer as a json
                    conversations = await response_get.json()
                except aiohttp.client_exceptions.ContentTypeError as e:
                    # Parsing failed
                    # Get the raw response text
                    response_text = await response_get.text()
                    # Displays (for logs) an error message with response details
                    raise RuntimeError(
                        f"ContentTypeError when trying to parse JSON from the response.\n"
                        f"Status: {response_get.status}, Content-Type: {response_get.headers.get('Content-Type')}\n"
                        f"Response body:\n{response_text}")

                answer = conversations[-1]
                if answer['role'] == 'assistant':
                    return answer
                else:
                    await asyncio.sleep(refresh_time)

async def client_task(client_id, oauth_client, nb_messages=10):
    """
    Simulate a client sending a fixed question multiple times and receiving answers.

    :param client_id: An identifier for the client.
    :param oauth_client: The SFAPIOAuthClient to connect to the API
    :param nb_messages: The number of messages to send.
    """
    convo_id = f"CONVID_{client_id}"
    fixed_question = "How can I connect to NERSC?"
    async with aiohttp.ClientSession() as session:
        print(f"Started client {client_id}")
        messages = []
        for message_id in range(nb_messages):
            messages.append({'role': 'user', 'content': fixed_question})
            answer_message = await get_answer(session, oauth_client, convo_id, messages)
            messages.append(answer_message)
            # Display progress with truncated answers for brevity
            display_answer = answer_message['content'] if (len(answer_message['content']) < 10) else (answer_message['content'][:10] + "...")
            print(f"Client {client_id} received answer {message_id+1}/{nb_messages}: '{display_answer}'")

async def main(nb_clients=10, nb_messages=1):
    """
    Set up and run the chat sessions concurrently.

    :param nb_clients: The number of simulated clients.
    :param api_base_url: The base URL of the API.
    :param nb_messages: The number of messages each client will send.
    """
    # Create and start tasks for all clients
    oauth_client = SFAPIOAuthClient(api_base_url=API_BASE_URL, token_url=TOKEN_URL)
    tasks = [client_task(i, oauth_client, nb_messages) for i in range(nb_clients)]
    await asyncio.gather(*tasks)

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())
