import lmntfy
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", default="./data/docs", type=str, help="path to the NERSC documentation folder")
    parser.add_argument("--database_path", default="./data/database", type=str, help="path to load/save the database") 
    parser.add_argument("--build_database", default=False, type=bool, help="whether to rebuild database from the documentation")
    parser.add_argument("--use_test_questions", default=True, type=bool, help="whether to run on the test questions (for testing purpose)")
    args = parser.parse_args()
    return args

def main():
    # process command line arguments
    args= parse_args()
    docs_folder = args.docs_path
    database_path_prefix = args.database_path
    rebuild_database = args.build_database
    use_test_questions = args.use_test_questions

    # initializes models
    llm = lmntfy.models.llm.GPT35()
    embedder = lmntfy.models.embedding.SBERTEmbedding()
    database = lmntfy.database.FaissDatabase(embedder)

    # initializes database
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

    # answers questions
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
