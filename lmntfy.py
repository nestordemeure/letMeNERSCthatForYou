import os
os.environ["TOKENIZERS_PARALLELISM"]="False"
import lmntfy
import json
import argparse
absolute_path = os.path.dirname(__file__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="/docs", type=str, help="where the documentation from NERSC is stored")
    parser.add_argument("--build_database", default=False, type=bool, help="whether to build database from docs")
    parser.add_argument("--use_test_questions", default=False, type=bool, help="whether to use test questions")
    parser.add_argument("--database", default="/database", type=str, help="documentation databse path") 
    args = parser.parse_args()
    return args

def main():
    args= parse_args()
    docs_folder = args.docs
    database_path_prefix = absolute_path + args.database
    rebuild_database = args.build_database
    use_test_questions = args.use_test_questions
    llm = lmntfy.models.llm.GPT35()
    embedder = lmntfy.models.embedding.SBERTEmbedding()
    database = lmntfy.database.FaissDatabase(embedder)
    if rebuild_database:
        print("Rebuilding the database.")
    

        document_loader = lmntfy.document_loader.DocumentLoader(llm)
        document_loader.load_folder(docs_folder, verbose=True)
        chunks = document_loader.documents()
        print(f"Produced {len(chunks)} chunks of maximum size {document_loader.max_chunk_size} tokens.")

        database.concurrent_add_chunks(chunks, verbose=True)
        print("Saved all chunks to the database.")

        database.save_to_file(database_path_prefix)
        print("Saved database to file.")
    else:
        print("Answering questions.")

        database.load_from_file(database_path_prefix)
        print("Loaded database from file.")

    question_answerer = lmntfy.QuestionAnswerer(llm, embedder, database)
    if use_test_questions:
        test_questions = ["What is NERSC?", "How can I connect to Perlmutter?", "Where do I find gcc?", "How do I kill all of my jobs?", "How can I run a job on GPU?"]
        for question in test_questions:
            print(f"\n> {question}\n")
            answer = question_answerer.get_answer(question, verbose=False)
            print(f"\n{answer}\n")
    else:
        messages = []
        while True:
            question = input("\n> ")
            messages.append({'role':'user', 'content': question})
            answer_message = question_answerer.continue_chat(messages, verbose=False)
            messages.append(answer_message)
            print(f"\n{answer_message['content']}")
 
if __name__ == "__main__":
    main()
