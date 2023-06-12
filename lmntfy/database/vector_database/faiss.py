import faiss
import json
import numpy as np
from . import VectorDatabase
from pathlib import Path

class FaissVectorDatabase(VectorDatabase):
    def __init__(self, embedding_length, name='faiss'):
        super().__init__(embedding_length, name)
        # vector database that will be used to store the vectors
        raw_index = faiss.IndexFlatIP(embedding_length)
        # index on top of the databse to support addition and deletion by id
        self.index = faiss.IndexIDMap(raw_index)
        self.current_id = 0

    def add(self, embedding):
        assert (embedding.size == self.embedding_length), "Invalid shape for embedding"
        self.index.add_with_ids(embedding, np.array([self.current_id], dtype=np.int64))
        self.current_id += 1
        return self.current_id - 1  # Return the ID of the added vector

    def remove_several(self, indices):
        self.index.remove_ids(np.array(indices, dtype=np.int64))

    def get_closest(self, input_embedding, k=3):
        assert (input_embedding.size == self.embedding_length), "Invalid shape for input_embedding"
        _, indices = self.index.search(input_embedding, k)
        return indices.flatten()

    def exists(self, database_folder:Path):
        """returns True if the database already exists on disk"""
        index_file = database_folder / 'index.faiss'
        database_data_file = database_folder / 'faiss_vector_database.json'
        return index_file.exists() and database_data_file.exists()

    def save(self, database_folder:Path):
        # saves the index
        index_path = database_folder / 'index.faiss'
        faiss.write_index(self.index, str(index_path))
        # saves the database data
        with open(database_folder / 'faiss_vector_database.json', 'w') as f:
            database_data = {'current_id': self.current_id}
            json.dump(database_data, f)

    def load(self, database_folder:Path):
        # load the index
        index_path = database_folder / 'index.faiss'
        self.index = faiss.read_index(str(index_path))
        # load the database_data
        with open(database_folder / 'faiss_vector_database.json', 'r') as f:
            database_data = json.load(f)
            self.current_id = database_data['current_id']