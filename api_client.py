"""
Basic shell client for the NERSC ai API

You will need to [store API credentials to be able to use the API](https://nersc.github.io/sfapi_client/quickstart/#:~:text=.perlmutter)-,Storing%20keys%20in%20files,-Keys%20can%20also)
"""
from sfapi_client import Client
from rich.markdown import Markdown
from rich.console import Console
import json
import time
import lmntfy

# use the dev side of the API
API_BASE_URL='https://api-dev.nersc.gov/api/internal/v1.2'
TOKEN_URL='https://oidc-dev.nersc.gov/c2id/token'

def get_answer(client:Client, convo_id, messages, refresh_time:int=1):
    # url to the ai API
    url=f'ai/docs?convo_id={convo_id}'

    # posts the conversation to the API
    response_post = client.post(url=url, data=json.dumps(messages))

    # returns the error message in case of failure
    if not response_post.is_success:
        return {'role':'assistant', 'content':response_post.text}

    # waits for the answer to arrive
    while True:
        answer = client.get(url).json()[-1]
        if answer['role'] == 'assistant':
            return answer
        else:
            time.sleep(refresh_time)

# NOTE: using the dev version of the API
with Client(api_base_url=API_BASE_URL, token_url=TOKEN_URL) as client:
    convo_id = f"CONVID" # TODO use a random number there
    lmntfy.user_interface.command_line.display_logo()

    messages = []
    console = Console()
    print()
    while True:
        # gets question
        question = input("> ")
        messages.append({'role':'user', 'content': question})
        # gets an answer and stores it
        answer_message = get_answer(client, convo_id, messages)
        messages.append(answer_message)
        # pretty prints the answer
        markdown_answer = Markdown(answer_message['content'])
        print()
        console.print(markdown_answer)
        print()