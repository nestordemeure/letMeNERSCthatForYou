import faiss
import json
import numpy as np
from . import Database

class FaissDatabase(Database):
    def __init__(self, embedder, file_path=None):
        super().__init__(embedder, file_path)
        self.index = faiss.IndexFlatIP(embedder.output_dim)
        self.documents = []

    def add_document(self, content, source):
        # Generate document embedding and add to the FAISS index
        embedding = np.array([self.embedder.embed(content)], dtype='float32')
        # Normalize the embedding
        if not self.embedder.normalized:
            faiss.normalize_L2(embedding)
        self.index.add(embedding)
        # Store the original document and its source
        self.documents.append({
            'content': content,
            'source': source,
            'embedding': embedding.tolist()  # Convert numpy array to list for json serialization
        })

    def get_closest_documents(self, input_text, k=1):
        # Generate input text embedding
        input_embedding = np.array([self.embedder.embed(input_text)], dtype='float32')
        # Normalize the embedding
        if not self.embedder.normalized:
            faiss.normalize_L2(input_embedding)
        # Query the FAISS index
        _, indices = self.index.search(input_embedding, k)
        # Return the corresponding documents
        return [self.documents[i] for i in indices.flatten()]

    def save_to_file(self, file_path):
        # Write FAISS index
        faiss.write_index(self.index, file_path + '_index.faiss')
        # Write documents data
        with open(file_path + '_docs.json', 'w') as f:
            json.dump(self.documents, f)

    def load_from_file(self, file_path):
        # Read FAISS index
        self.index = faiss.read_index(file_path + '_index.faiss')
        # Read documents data
        with open(file_path + '_docs.json', 'r') as f:
            self.documents = json.load(f)
