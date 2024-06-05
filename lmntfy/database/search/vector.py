import faiss
import numpy as np
from functools import partial
from pathlib import Path
from typing import List
from ..chunk import Chunk
from . import SearchEngine
from ...models.embedding import Embedding
from ..document_splitter import chunk_splitter

class VectorSearch_raw(SearchEngine):
    """
    Sentence-embedding based vector search.
    Based on [faiss](https://faiss.ai/).
    """
    def __init__(self, embedder: Embedding, name:str='vector'):
        # embedder
        self.embedder: Embedding = embedder
        # vector database that will be used to store the vectors
        raw_index = faiss.IndexFlatIP(embedder.embedding_length)
        # index on top of the database to support addition and deletion by id
        self.index = faiss.IndexIDMap(raw_index)
        # init parent
        super().__init__(name=name + embedder.name)

    def _add_chunk(self, chunk_id:int, chunk: Chunk):
        """
        Adds a chunk with the given id.
        NOTE: assumes that the chunk is small enough to fit the context length.
        """
        # embedds the input
        content_embedding = self.embedder.embed(chunk.content, is_query=False)
        # create a single element batch with the embeddings and indices
        embedding_batch = content_embedding.reshape((1,-1))
        id_batch = np.array([chunk_id], dtype=np.int64)
        # adds them to the vector database
        self.index.add_with_ids(embedding_batch, id_batch)

    def add_several_chunks(self, chunks: dict[int,Chunk]):
        """
        Adds several chunks with the given indices.
        NOTE: breaks chunk down into subchunks that fit our embedding model's context length
        """
        for (chunk_id, chunk) in chunks:
            # breaks the chunk into subchunks small enough to fit the embedder's context size
            subchunks = chunk_splitter(chunk, self.embedder.count_tokens, self.embedder.context_size)
            # adds them one at a time, all pointing to the same chunk_id (parent document retrieval)
            for subchunk in subchunks:
                self._add_chunk(chunk_id, subchunk)

    def remove_several_chunks(self, chunk_indices: List[int]):
        """
        Removes several chunks from the search engine.
        """
        self.index.remove_ids(np.array(chunk_indices, dtype=np.int64))
    
    def get_closest_chunks(self, input_text: str, k: int) -> List[(float,int)]:
        """
        Returns the (score,chunk_id) of the closest chunks, from best to worst
        """
        # embedds the input
        input_embedding = self.embedder.embed(input_text, is_query=True)
        # reshape it ino a batch of size one
        input_embedding_batch = input_embedding.reshape((1,-1))
        # does the search
        distances, indices = self.index.search(input_embedding_batch, k=k)
        # zip the results into a single list
        distances = distances.flatten().tolist()
        indices = indices.flatten().tolist()
        return list(zip(distances, indices))

    def save(self, database_folder:Path):
        """
        Save the search engine on file.
        """
        index_path = database_folder / 'index.faiss'
        faiss.write_index(self.index, str(index_path))

    def load(self, database_folder:Path):
        """
        Loads the search engine from file.
        """
        index_path = database_folder / 'index.faiss'
        self.index = faiss.read_index(str(index_path))

# instance that lets you define the embedder
VectorSearch = lambda embedder: partial(VectorSearch_raw, embedder=embedder)