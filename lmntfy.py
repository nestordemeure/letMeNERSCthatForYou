import lmntfy
from pathlib import Path

#------------------------------------------------------------------------------
# PARAMETERS

data_folder = Path('./data')
docs_folder = data_folder / 'docs_lite'

llm = lmntfy.models.llm.chatgpt.GPT35()
embedder = lmntfy.models.embedding.openai_embedding.OpenAIEmbedding()

#------------------------------------------------------------------------------
# LOAD DOCUMENTS

document_loader = lmntfy.document_loader.DocumentLoader(llm)
document_loader.load_folder(docs_folder, verbose=True)
chunks = document_loader.documents()

print(f"Done! Produced {len(chunks)} chunks of maximum size {document_loader.max_chunk_size} tokens.")

#for chunk in chunks:
#    print(f"source:{chunk['source']} content:\n{chunk['content']}\n")