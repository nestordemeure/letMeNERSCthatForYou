import lmntfy
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--logs_folder", default=None, type=Path, help="path to the log saving folder") 
    parser.add_argument("--update_database", default=False, action='store_true', help="whether to update database to the current documentation")
    parser.add_argument("--use_test_questions", default=False, action='store_true', help="whether to run on the test questions (for debugging purposes)")
    args = parser.parse_args()
    return args

def main():
    # process command line arguments
    args= parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    logs_folder = args.logs_folder
    update_database = args.update_database
    use_test_questions = args.use_test_questions

    # initializes models
    print("Loading the database and models...")
    llm = lmntfy.models.llm.GPT35()
    embedder = lmntfy.models.embedding.SBERTEmbedding()
    database = lmntfy.database.FaissDatabase(llm, embedder, docs_folder, database_folder, update_database=update_database)
    question_answerer = lmntfy.QuestionAnswerer(llm, embedder, database, logs_folder=logs_folder)

    # answers questions
    lmntfy.user_interface.command_line.display_logo()
    if use_test_questions:
        # run on a handful of test question for quick evaluation purposes
        test_questions = ["What is NERSC?", "How do I use sshproxy?", "How can I connect to Perlmutter?", "Where do I find gcc?", "How do I kill all of my jobs?", "How can I run a job on GPU?"]
        lmntfy.user_interface.command_line.answer_questions(question_answerer, test_questions)
    else:
        # chat with the model
        lmntfy.user_interface.command_line.chat(question_answerer)
 
if __name__ == "__main__":
    main()
