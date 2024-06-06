"""
Runs async queries against a local chatbot to ensure it can deal with concurency properly.
"""
import lmntfy
import asyncio
import argparse
from pathlib import Path
from lmntfy.question_answering import QuestionAnswerer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder",default="../models", type=Path, help="path to the folder containing all the models")
    args = parser.parse_args()
    return args

# Global lock to ensure thread safety in async environments
chat_lock = asyncio.Lock()

async def client_task(question_answerer: QuestionAnswerer, client_id:int, nb_messages:int=10):
    """
    Simulate a client sending a fixed question and receiving answers in a loop.

    :param question_answerer: the model to which questions are asked
    :param client_id: An identifier for the client.
    :param nb_messages: The number of messages to send.
    """
    print(f"Started client {client_id}")
    messages = []
    fixed_question = "How can I connect to NERSC?"
    for message_id in range(1, nb_messages+1):
        # gets answer from the model
        print(f"Client {client_id} sending question {message_id}/{nb_messages}...")
        messages.append({'role': 'user', 'content': fixed_question})
        
        # raw, sync, call
        #answer_message = question_answerer.answer_messages(messages) # TODO async_chat?

        # pseudo async, one at a time in a thread
        #async with chat_lock:
        #    answer_message = await asyncio.to_thread(question_answerer.chat, messages)

        answer_message = await question_answerer.answer_messages(messages)

        messages.append(answer_message)
        # Display progress with truncated answers for brevity
        display_answer = answer_message['content'] if (len(answer_message['content']) < 10) else (answer_message['content'][:10] + "...")
        print(f"Client {client_id} received answer {message_id}/{nb_messages}: '{display_answer}'")

async def main(nb_clients:int=10, nb_messages:int=10):
    """
    Set up and run chat sessions concurrently.

    :param nb_clients: The number of simulated clients.
    :param nb_messages: The number of messages each client will send.
    """
    # display the logo for esthetic reasons
    lmntfy.user_interface.command_line.display_logo()

    # process command line arguments
    args= parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    models_folder = args.models_folder

    # initializes models
    print("Loading the database and models...")
    search_engine = lmntfy.database.search.Default(models_folder, device='cuda')
    llm = lmntfy.models.llm.Default(models_folder, device='cuda')
    database = lmntfy.database.Database(docs_folder, database_folder, search_engine, llm, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, database)

    # Create and start tasks for all clients
    tasks = [client_task(question_answerer, i, nb_messages) for i in range(1, nb_clients+1)]
    await asyncio.gather(*tasks)
    return

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())
