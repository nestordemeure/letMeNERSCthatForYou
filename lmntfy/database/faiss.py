import faiss
import json
import numpy as np
from . import Database

class FaissDatabase(Database):
    def __init__(self, embedder):
        super().__init__(embedder)
        self.index = faiss.IndexFlatIP(embedder.output_dim)
        self.chunks = []

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

    @staticmethod
    def load_from_file(file_path, embedder):
        db = FaissDatabase(embedder)
        # Read FAISS index
        db.index = faiss.read_index(file_path + '_index.faiss')
        # Read chunks data
        with open(file_path + '_chunks.json', 'r') as f:
            db.chunks = json.load(f)
        return db
