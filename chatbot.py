import lmntfy
import argparse
from pathlib import Path

def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder",default="../models",type=Path, help="path to the folder containing all the models")
    parser.add_argument("--logs_folder", default=None, type=Path, help="path to the log saving folder") 
    parser.add_argument("question", nargs="*", default=[], help="optional question passed to the script")
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
    logs_folder = args.logs_folder
    question = args.question

    # initializes models
    llm = lmntfy.models.llm.Vicuna(models_folder)
    embedder = lmntfy.models.embedding.SBERTEmbedding(models_folder)
    database = lmntfy.database.FaissDatabase(llm, embedder, docs_folder, database_folder, update_database=False)
    question_answerer = lmntfy.QuestionAnswerer(llm, embedder, database, logs_folder=logs_folder)

    # answers questions
    if len(question) > 0:
        # run on a handful of test question for quick evaluation purposes
        lmntfy.user_interface.command_line.answer_question(question_answerer, question)
    else:
        # chat with the model
        lmntfy.user_interface.command_line.display_logo()
        lmntfy.user_interface.command_line.chat(question_answerer, verbose=True)
 
if __name__ == "__main__":
    main()
