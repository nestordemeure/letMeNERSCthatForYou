import lmntfy
import argparse
import asyncio
from pathlib import Path

def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder",default="../models",type=Path, help="path to the folder containing all the models")
    parser.add_argument("question", nargs="*", default=[], help="optional question passed to the script")
    args = parser.parse_args()
    # Convert the question list back to a string
    args.question = " ".join(args.question).strip()
    return args

async def main():
    # process command line arguments
    args = parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    models_folder = args.models_folder
    question = args.question

    # initializes models
    search_engine = lmntfy.database.search.Default(models_folder, device='cuda')
    llm = lmntfy.models.llm.Default(models_folder, device='cuda')
    database = lmntfy.database.Database(docs_folder, database_folder, search_engine, llm, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, database)

    # answers questions
    if len(question) > 0:
        # run on a handful of test question for quick evaluation purposes
        await lmntfy.user_interface.command_line.answer_question(question_answerer, question)
    else:
        # chat with the model
        lmntfy.user_interface.command_line.display_logo()
        await lmntfy.user_interface.command_line.chat(question_answerer, verbose=False)
 
if __name__ == "__main__":
    asyncio.run(main())
