import lmntfy
import argparse
import asyncio
from pathlib import Path
from lmntfy.models.llm.engine import VllmEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder",default="../models", type=Path, help="path to the folder containing all the models")
    parser.add_argument("--update_database", default=False, action='store_true', help="whether to update database to the current documentation")
    parser.add_argument("--use_test_questions", default=True, action='store_true', help="whether to run on the test questions (for debugging purposes)")
    parser.add_argument("--debug",default=False,action="store_true",help="Print useful debug information (e.g., prompts)",)
    args = parser.parse_args()
    return args

async def main():
    # process command line arguments
    args = parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    models_folder = args.models_folder
    update_database = args.update_database
    use_test_questions = args.use_test_questions

    # initializes models
    print("Loading the database and models...")
    search_engine = lmntfy.database.search.Default(models_folder, device='cuda')
    llm = lmntfy.models.llm.Default(models_folder, device='cuda', engineType=VllmEngine)
    database = lmntfy.database.Database(docs_folder, database_folder, search_engine, llm, update_database=update_database)
    question_answerer = lmntfy.QuestionAnswerer(llm, database)

    # answers questions
    lmntfy.user_interface.command_line.display_logo()
    if use_test_questions:
        # run on a handful of test question for quick evaluation purposes
        test_questions = ["What is JAX?",
                          "Where do I find gcc?", 
                          "How do I use sshproxy?", 
                          "How can I connect to Perlmutter?", 
                          "How do I kill all of my jobs?", 
                          "How can I run a job on GPU?",
                          "What is the meaning of life?",
                          "What is NERSC?"]
        await lmntfy.user_interface.command_line.answer_questions(question_answerer, test_questions, verbose=False)
    else:
        # chat with the model
        await lmntfy.user_interface.command_line.chat(question_answerer, verbose=False)

if __name__ == "__main__":
    asyncio.run(main())
