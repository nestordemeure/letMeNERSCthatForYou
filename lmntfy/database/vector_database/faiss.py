import faiss
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
        # TODO can we simplify? 
        # self.index.add_with_ids(embedding, np.array([self.current_id], dtype=np.int64))
        self.index.add_with_ids(embedding, self.current_id)
        self.current_id += 1
        return self.current_id - 1  # Return the ID of the added vector

    def remove_several(self, indices):
        self.index.remove_ids(np.array(indices, dtype=np.int64))

    def get_closest(self, input_embedding, k=3):
        assert (input_embedding.size == self.embedding_length), "Invalid shape for input_embedding"
        _, indices = self.index.search(input_embedding, k)
        return indices.flatten()

    def save(self, file_path:Path):
        faiss.write_index(self.index, str(file_path))

    def load(self, file_path:Path):
        self.index = faiss.read_index(str(file_path))
        self.current_id = self.index.ntotal
