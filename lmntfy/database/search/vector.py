import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from ..chunk import Chunk
from . import SearchEngine
from ...models.embedding import Embedding
from ..document_splitter import chunk_splitter
from .hybrid import merge_and_sort_scores

class VectorSearch(SearchEngine):
    """
    Sentence-embedding based vector search.
    Based on [faiss](https://faiss.ai/).
    """
    def __init__(self, embedder: Embedding, max_tokens_per_chunk:int=None):
        """
        embedder (Embedding): the model used to compute the embeddings
        max_tokens_per_chunk (optional int): the maximum size for the chunks (default/capped to embedder.context_size)
        """
        # embedder
        self.embedder: Embedding = embedder
        self.max_tokens_per_chunk = self.embedder.context_size if (max_tokens_per_chunk is None) else min(max_tokens_per_chunk, self.embedder.context_size)
        # vector database that will be used to store the vectors
        raw_index = faiss.IndexFlatIP(embedder.embedding_length)
        # index on top of the database to support addition and deletion by id
        self.index = faiss.IndexIDMap(raw_index)
        # init parent
        super().__init__(name=f"vector-{embedder.name}-{self.max_tokens_per_chunk}")

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

    def add_several_chunks(self, chunks: Dict[int,Chunk], verbose=True):
        """
        Adds several chunks with the given indices.
        NOTE: breaks chunk down into subchunks that fit our embedding model's context length
        """
        for (chunk_id, chunk) in tqdm(chunks.items(), disable=not verbose, desc="Vector embedding chunks"):
            # breaks the chunk into subchunks small enough to fit the embedder's context size
            subchunks = chunk_splitter(chunk, self.embedder.count_tokens, self.max_tokens_per_chunk)
            # adds them one at a time, all pointing to the same chunk_id (parent document retrieval)
            for subchunk in subchunks:
                self._add_chunk(chunk_id, subchunk)

    def remove_several_chunks(self, chunk_indices: List[int]):
        """
        Removes several chunks from the search engine.
        """
        self.index.remove_ids(np.array(chunk_indices, dtype=np.int64))
    
    def get_closest_chunks(self, input_text: str, chunks:Dict[int,Chunk], k: int) -> List[Tuple[float,int]]:
        """
        Returns the (score,chunk_id) of the closest chunks, from best to worst
        """
        # embedds the input
        input_embedding = self.embedder.embed(input_text, is_query=True)
        # reshape it into a batch of size one
        input_embedding_batch = input_embedding.reshape((1,-1))
        # loop until we get enough items
        # NOTE: due to several ids pointing to the same chunk, we migh get duplicates
        similarities = list()
        indices = list()
        k_queried = k
        while len(set(indices)) < k:
            # does the search
            similarities, indices = self.index.search(input_embedding_batch, k=k_queried)
            similarities = similarities.flatten().tolist()
            indices = indices.flatten().tolist()
            k_queried *= 2
        # zip the results into a single list and remove duplicates
        scored_chunkids = list(zip(similarities, indices))
        return merge_and_sort_scores(scored_chunkids, merging_strategy=max)

    def initialize(self, database_folder:Path):
        """
        Initialize the search engine if needed.
        """
        # no initializaion needed
        return

    def exists(self, database_folder:Path) -> bool:
        """
        Returns True if an instance of the search engine is saved in the given folder.
        """
        index_path = database_folder / 'index.faiss'
        return index_path.exists()

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
