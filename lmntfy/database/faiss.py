import faiss
import json
import numpy as np
from . import Database

class FaissDatabase(Database):
    def __init__(self, embedder):
        super().__init__(embedder)
        self.index = faiss.IndexFlatIP(embedder.embedding_length)
        self.chunks = []

    def concurrent_add_chunks(self, chunks, verbose=False):
        print(f"WARNING: FAISS does not support `concurrent_add_chunks`, defaulting to `add_chunks`")
        return self.add_chunks(chunks, verbose=verbose)

    def add_chunk(self, chunk):
        # Generate chunk embedding and add to the FAISS index
        embedding = np.array([self.embedder.embed(chunk['content'])], dtype='float32')
        self.index.add(embedding)
        # Store the original chunk
        self.chunks.append(chunk)

    def get_closest_chunks(self, input_text, k=3):
        if len(self.chunks) <= k: return self.chunks
        # Generate input text embedding
        input_embedding = np.array([self.embedder.embed(input_text)], dtype='float32')
        # Query the FAISS index
        _, indices = self.index.search(input_embedding, k)
        # Return the corresponding chunks
        return [self.chunks[i] for i in indices.flatten()]

    def save_to_file(self, file_path):
        # Write FAISS index
        faiss.write_index(self.index, file_path + '_index.faiss')
        # Write chunks data
        with open(file_path + '_chunks.json', 'w') as f:
            json.dump(self.chunks, f)

    def load_from_file(self, file_path):
        # Read FAISS index
        self.index = faiss.read_index(file_path + '_index.faiss')
        # Read chunks data
        with open(file_path + '_chunks.json', 'r') as f:
            self.chunks = json.load(f)