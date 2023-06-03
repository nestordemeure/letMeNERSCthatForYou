from . import Database
from scipy.spatial import distance
import numpy as np
import json

class NaiveDatabase(Database):
    def __init__(self, embedder):
        super().__init__(embedder)
        self.chunks = []

    def add_chunk(self, chunk):
        embedding = self.embedder.embed(chunk['content'])
        self.chunks.append({
            'chunk': chunk,
            'embedding': embedding
        })

    def get_closest_chunks(self, input_text, k=3):
        if len(self.chunks) <= k: return self.chunks
        # Calculate cosine distances to the input embedding and return the k documents with smallest distance
        input_embedding = self.embedder.embed(input_text)
        distances = [distance.cosine(input_embedding, chunk['embedding']) for chunk in self.chunks]
        closest_indices = np.argsort(distances)[:k]
        return [self.chunks[i]['chunk'] for i in closest_indices]

    def save_to_file(self, file_path):
        # Convert numpy arrays to lists before saving
        with open(file_path + '_chunks.json', 'w') as f:
            json.dump(self.chunks, f)

    @staticmethod
    def load_from_file(file_path, embedder):
        db = NaiveDatabase(embedder)
        with open(file_path + '_chunks.json', 'r') as f:
            db.chunks = json.load(f)
        return db