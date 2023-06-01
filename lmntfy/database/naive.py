from . import Database
from scipy.spatial import distance
import numpy as np
import json

class NaiveDatabase(Database):
    def __init__(self, embedder, file_path=None):
        self.documents = []
        super().__init__(embedder, file_path)

    def add_document(self, content, source):
        embedding = self.embedding.embed(content)
        self.documents.append({
            'content': content,
            'source': source,
            'embedding': embedding
        })

    def get_closest_documents(self, input_text, k=1):
        input_embedding = self.embedding.embed(input_text)
        # Calculate cosine distances to the input embedding and return the k documents with smallest distance
        distances = [distance.cosine(input_embedding, doc['embedding']) for doc in self.documents]
        closest_indices = np.argsort(distances)[:k]
        return [self.documents[i] for i in closest_indices]

    def save_to_file(self, file_path):
        # Convert numpy arrays to lists before saving
        documents_as_lists = [{**doc, 'embedding': doc['embedding'].tolist()} for doc in self.documents]
        with open(file_path, 'w') as f:
            json.dump(documents_as_lists, f)

    def load_from_file(self, file_path):
        with open(file_path, 'r') as f:
            # Convert lists back to numpy arrays after loading
            documents_as_lists = json.load(f)
            self.documents = [{**doc, 'embedding': np.array(doc['embedding'])} for doc in documents_as_lists]
