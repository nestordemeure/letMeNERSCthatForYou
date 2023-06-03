import lmntfy

#------------------------------------------------------------------------------
# PARAMETERS

data_folder = './data'
docs_folder = data_folder + '/docs_lite'
database_path_prefix = data_folder + '/database'

llm = lmntfy.models.llm.GPT35()
embedder = lmntfy.models.embedding.OpenAIEmbedding()
database = lmntfy.database.NaiveDatabase(embedder)

#------------------------------------------------------------------------------
# LOAD DOCUMENTS

document_loader = lmntfy.document_loader.DocumentLoader(llm)
document_loader.load_folder(docs_folder, verbose=True)
chunks = document_loader.documents()

print(f"Produced {len(chunks)} chunks of maximum size {document_loader.max_chunk_size} tokens.")

#------------------------------------------------------------------------------
# SAVE CHUNKS IN DATABASE

database.concurent_add_chunks(chunks)
print("Saved all chunks to the database.")

#database.save_to_file(database_path_prefix)
#print("Saved database to file.")

#new_database = lmntfy.database.NaiveDatabase.load_from_file(database_path_prefix, embedder)
#print("Loaded database from file.")

demo_text = chunks[42]['content']
closests = database.get_closest_chunks(demo_text)
print(f"\nINPUT:\n{demo_text}\n")
for chunk in closests:
    print(f"\nSOURCE:{chunk['source']}\n{chunk['content']}\n")

#------------------------------------------------------------------------------
print("Done!")