#!/global/cfs/cdirs/nstaff/chatbot/conda/bin/python3
"""
API stress test with multiple clients chatting simultaneously.

This script is designed to simulate multiple clients sending messages to an API endpoint concurrently.
It uses asynchronous programming to handle multiple chat sessions in parallel, each repeating a fixed question multiple times.
"""
import aiohttp
import asyncio

async def get_answer(session, convo_id, messages, api_base_url, refresh_time: int=1):
    """
    Send messages to the API and wait for the answer asynchronously.

    :param session: The aiohttp session object for making HTTP requests.
    :param convo_id: A unique conversation identifier.
    :param messages: A list of messages (questions) sent in the conversation.
    :param api_base_url: The base URL of the API.
    :param refresh_time: Time interval to wait before polling the API for an answer.
    :return: The answer received from the assistant.
    """
    url = f'{api_base_url}/ai/docs?convo_id={convo_id}'
    
    # Post the conversation messages to the API and check the response
    async with session.post(url, json=messages) as response_post:
        if (response_post.status != 200):
            text = await response_post.text()
            print(f"ERROR (status:{response_post.status}): {text}")
            return {'role': 'assistant', 'content': text}

        # Poll the API for an answer and return it when received
        while True:
            async with session.get(url) as response_get:
                answer = (await response_get.json())[-1]
                if answer['role'] == 'assistant':
                    return answer
                else:
                    await asyncio.sleep(refresh_time)

async def client_task(client_id, api_base_url, convo_id, nb_messages=10):
    """
    Simulate a client sending a fixed question multiple times and receiving answers.

    :param client_id: An identifier for the client.
    :param api_base_url: The base URL of the API.
    :param convo_id: A unique conversation identifier.
    :param nb_messages: The number of messages to send.
    """
    fixed_question = "How can I connect to NERSC?"
    async with aiohttp.ClientSession() as session:
        messages = []
        for message_id in range(nb_messages):
            messages.append({'role': 'user', 'content': fixed_question})
            answer_message = await get_answer(session, convo_id, messages, api_base_url)
            messages.append(answer_message)
            # Display progress with truncated answers for brevity
            display_answer = answer_message['content'] if (len(answer_message['content']) < 10) else (answer_message['content'][:10] + "...")
            print(f"Client {client_id} received answer {message_id}/{nb_messages}: '{display_answer}'")

async def main(nb_clients=200, api_base_url='https://api-dev.nersc.gov/api/v1.2', nb_messages=10):
    """
    Set up and run the chat sessions concurrently.

    :param nb_clients: The number of simulated clients.
    :param api_base_url: The base URL of the API.
    :param nb_messages: The number of messages each client will send.
    """
    # Create and start tasks for all clients
    tasks = [client_task(i, api_base_url, f"CONVID_{i}", nb_messages) for i in range(nb_clients)]
    await asyncio.gather(*tasks)

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())
