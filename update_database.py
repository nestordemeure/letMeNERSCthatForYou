"""
This script updates the database then stops.
"""
import lmntfy
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder", default="./data/docs", type=Path, help="path to the NERSC documentation folder")
    parser.add_argument("--database_folder", default="./data/database", type=Path, help="path to the database saving folder") 
    args = parser.parse_args()
    return args

def main():
    # process command line arguments
    args= parse_args()
    docs_folder = args.docs_folder
    database_folder = args.database_folder

    # load the database and updates it if needed
    llm = lmntfy.models.llm.GPT35()
    embedder = lmntfy.models.embedding.SBERTEmbedding()
    database = lmntfy.database.FaissDatabase(llm, embedder, docs_folder, database_folder, update_database=True)
    print("Done!")

if __name__ == "__main__":
    main()
