import lmntfy
import json

#------------------------------------------------------------------------------
# PARAMETERS

data_folder = './data'
docs_folder = data_folder + '/docs'
database_path_prefix = data_folder + '/database'
rebuild_database = False
use_test_questions = False

llm = lmntfy.models.llm.GPT35()
embedder = lmntfy.models.embedding.OpenAIEmbedding()
database = lmntfy.database.FaissDatabase(embedder)

#------------------------------------------------------------------------------
# MAIN

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
        while True:
            question = input("\n> ")
            answer = question_answerer.get_answer(question, verbose=False)
            print(f"\n{answer}")