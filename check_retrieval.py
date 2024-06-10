"""
This script runs the retriveal pipeline for various questions in order to see how well it does.
"""
import lmntfy
import argparse
from pathlib import Path
from lmntfy.models.llm.engine import TransformerEngine

# questions that will be used to test the retrieval
TEST_QUESTIONS = [
    "How can I connect to NERSC?",
    "How do I use JAX at NERSC?",
    "How can I kill all my jobs?",
    "How can I run a cron job on Perlmutter?",
    "Where is gcc?",
    "How do I use sshproxy?"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/nersc_doc/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    parser.add_argument("--models_folder",default="../models",type=Path, help="path to the folder containing all the models")
    args = parser.parse_args()
    return args

def main():
    # process command line arguments
    args= parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder
    models_folder = args.models_folder

    # initializes models
    search_engine = lmntfy.database.search.Default(models_folder, device='cuda')
    llm = lmntfy.models.llm.Default(models_folder, device='cpu', engineType=TransformerEngine)
    database = lmntfy.database.Database(docs_folder, database_folder, 
                                        search_engine, llm, 
                                        update_database=False)

    # runs the retrieval with `verbose` on
    for question in TEST_QUESTIONS:
        database.get_closest_chunks(question, k=8, verbose=True)

if __name__ == "__main__":
    main()
