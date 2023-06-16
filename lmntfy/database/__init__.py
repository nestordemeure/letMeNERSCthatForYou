import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from abc import ABC, abstractmethod
from ..models import LanguageModel, Embedding
from .document_loader import Chunk, path2url, chunk_file
from .file import File
from .document_loader.token_count_pair import TokenCountPair

class Database(ABC):
    def __init__(self, llm:LanguageModel, embedder:Embedding,
                       documentation_folder:Path, database_folder:Path, 
                       min_chunks_per_query=3, update_database=True, name=None):
        self.name = name
        self.embedder = embedder
        self.embedding_length = embedder.embedding_length
        # maximum size of each chunk
        # we leave space for two additional chunks, representing the prompt and the model's answer
        llm_max_tokens = llm.context_size / (min_chunks_per_query + 2)
        self.max_tokens_per_chunk = TokenCountPair(llm_max_tokens, embedder.max_input_tokens)
        # the token counting function
        self.token_counter = TokenCountPair.build_pair_counter(llm.token_counter, embedder.token_counter)
        # dictionary of all files
        # file_path -> File
        self.files: Dict[Path, File] = dict()
        # dictionary of all chunk
        # vector_database_index -> Chunk
        self.chunks: Dict[int, Chunk] = dict()
        # loads the database from file if possible
        self.documentation_folder = documentation_folder
        self.database_folder = database_folder
        self.load(update_database=update_database, verbose=True)
    
    # ----- VECTOR DATABASE OPERATIONS -----

    @abstractmethod
    def _index_add(self, embedding: np.ndarray) -> int:
        """
        Abstract method for adding a new vector to the vector database
        returns its index
        """
        pass

    @abstractmethod
    def _index_remove_several(self, indices: List[int]):
        """
        Abstract method for removing vectors from the vector database
        """
        pass

    @abstractmethod
    def _index_get_closest(self, input_embedding: np.ndarray, k=3) -> List[int]:
        """
        Abstract method, returns the indices of the k closest embeddings in the vector database
        """
        pass

    def get_closest_chunks(self, input_text: str, k: int = 3) -> List[Chunk]:
        if len(self.chunks) <= k: return list(self.chunks.values())
        # Generate input text embedding
        input_embedding = self.embedder.embed(input_text)
        # Query the vector databse
        indices = self._index_get_closest(input_embedding, k)
        # Return the corresponding chunks
        return [self.chunks[i] for i in indices]

    # ----- FILE OPERATIONS -----

    def remove_file(self, file_path):
        """Removes a file's content from the Database"""
        file = self.files[file_path]
        indices_to_remove = file.vector_database_indices
        # remove file chunks from chunks
        for index in indices_to_remove:
            del self.chunks[index]
        # remove embeddings from vector database
        self._index_remove_several(indices_to_remove)
        # remove file from files
        del self.files[file_path]

    def add_file(self, file_path):
        """Add a file's content to the database"""
        # generates the url to the file
        url = path2url(file_path, self.documentation_folder)
        # slice file into chunks
        chunks = chunk_file(file_path, url, self.token_counter, self.max_tokens_per_chunk)
        # save chunks in the databse
        file_update_date = datetime.fromtimestamp(file_path.stat().st_mtime)
        file = File(creation_date=file_update_date)
        for chunk in chunks:
            # compute the embedding of the chunk
            embedding = self.embedder.embed(chunk.content)
            # add embedding to the vector database
            chunk_index = self._index_add(embedding)
            # add chunk to file
            file.add_index(chunk_index)
            # add chunk to chunks
            self.chunks[chunk_index] = chunk
        # add file to files
        self.files[file_path] = file

    def update(self, verbose=False):
        """Goes over the documentation and insures that we are up to date then saves the result."""
        # removes files that do not exist anymore or are out of date
        existing_files = list(self.files.items())
        for file_path, file in tqdm(existing_files, disable=not verbose, desc="Removing old files"):
            if (not file_path.exists()) or (datetime.fromtimestamp(file_path.stat().st_mtime) > file.creation_date):
                self.remove_file(file_path)
        # add new files
        current_files = [Path(root) / file for root, dirs, files in os.walk(self.documentation_folder) for file in files]
        if len(current_files) == 0:
            raise RuntimeError(f"ERROR: the documentation folder '{self.documentation_folder}' is empty or does not exist.")
        for file_path in tqdm(current_files, disable=not verbose, desc="Loading new files"):
            if not file_path in self.files:
                self.add_file(file_path)
        # save resulting database
        self.save()

    def exists(self):
        """returns True if the database already exists on disk"""
        files_file = self.database_folder / 'files.json'
        chunks_file = self.database_folder / 'chunks.json'
        return self.database_folder.exists() and files_file.exists() and chunks_file.exists()

    def load(self, update_database=True, verbose=False):
        if self.exists():
            # load the files info
            with open(self.database_folder / 'files.json', 'r') as f:
                files_dict = json.load(f)
                self.files = {Path(k): File.from_dict(v) for k, v in files_dict.items()}
            # load the chunks
            with open(self.database_folder / 'chunks.json', 'r') as f:
                chunks_dict = json.load(f)
                self.chunks = {int(k): Chunk.from_dict(v) for k, v in chunks_dict.items()}
        elif verbose:
            print(f"Warning: '{self.database_folder}' or its content does not currently exist. The database will be created from scratch.")
        # updates the database to the latest documentation
        if update_database or not self.exists():
            self.update(verbose=verbose)

    def save(self):
        # insures that the saving folder exists
        self.database_folder.mkdir(parents=True, exist_ok=True)
        # saves the files info
        with open(self.database_folder / 'files.json', 'w') as f:
            files_dict = {str(k): v.to_dict() for k, v in self.files.items()}
            json.dump(files_dict, f)
        # saves the chunks
        with open(self.database_folder / 'chunks.json', 'w') as f:
            chunks_dict = {k: v.to_dict() for k, v in self.chunks.items()}
            json.dump(chunks_dict, f)

from .faiss import FaissDatabase