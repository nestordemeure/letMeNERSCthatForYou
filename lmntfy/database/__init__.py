import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from abc import ABC, abstractmethod
from ..models import LanguageModel, Embedding
from .document_loader import Chunk, chunk_file
from .document_loader.markdown_spliter import markdown_splitter
from .file import File

class Database(ABC):
    def __init__(self, llm:LanguageModel, embedder:Embedding,
                       documentation_folder:Path, database_folder:Path, 
                       min_chunks_per_query=8, update_database=True, name=None):
        self.name = name
        self.embedder = embedder
        self.embedding_length = embedder.embedding_length
        # maximum size of each chunk
        # we leave space for two additional chunks, representing the prompt and the model's answer
        self.max_tokens_per_chunk = llm.context_size / (min_chunks_per_query + 2)
        # the token counting function of the llm
        self.count_tokens_llm = llm.count_tokens
        # dictionary of all files
        # file_path -> File
        self.files: Dict[Path, File] = dict()
        # dictionary of all chunk
        # vector_database_index -> Chunk
        # NOTE: it might contain identical chunks pointed at by different indices
        self.chunks: Dict[int, Chunk] = dict()
        # set of file and folder names that will not be processed
        self.ignored_files_and_folders: Set[str] = {'timeline'}
        # loads the database from file if possible
        self.documentation_folder = documentation_folder.absolute().resolve()
        self.database_folder = database_folder.absolute().resolve()
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
        """
        returns the (at most) k chunks that contains pieces of text closest to the input_text according to its embedding
        """
        if len(self.chunks) <= k: return list(self.chunks.values())
        # Generate input text embedding
        # WARNING: we assume that the input is small enough to be embedded
        input_embedding = self.embedder.embed(input_text)
        # Loop until we get enough distinct chunks
        # NOTE: identical chunks is *only* a risk when several indices might be pointing to (diferent parts of) the same chunk
        chunks = set()
        k_growing_factor = 1
        while len(chunks) < k:
            # Query the vector database
            indices = self._index_get_closest(input_embedding, k*k_growing_factor)
            # Gets the correspondings chunks ensuring uniqueness
            # NOTE: we build on Python sets preserving ordering past Python3.7
            chunks = {self.chunks[i] for i in indices}
            # If we fail, th next call will request twice as many items
            k_growing_factor *= 2
        # converts chunks into a properly sized list
        chunks = list(chunks)[:k]
        # Return the corresponding chunks
        return chunks

    # ----- FILE OPERATIONS -----

    def _is_ignored_file(self, file_path: Path) -> bool:
        """Returns true if the name of the file or of one of its parent folders is in self.ignored_files_and_folders"""
        return file_path.name in self.ignored_files_and_folders or \
               any(parent.name in self.ignored_files_and_folders for parent in file_path.parents)

    def remove_file(self, file_path:Path):
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

    def add_file(self, file_path:Path):
        """Add a file's content to the database"""
        # shortcut the function on ignored files
        if self._is_ignored_file(file_path): return
        # open file in the database
        file_update_date = datetime.fromtimestamp(file_path.stat().st_mtime)
        file = File(creation_date=file_update_date)
        # slice file into chunks small enough to fit several in the model's context size
        chunks = chunk_file(file_path, self.documentation_folder, self.count_tokens_llm, self.max_tokens_per_chunk)
        for chunk in chunks:
            # slice chunk into sub chunks small enough to be embedded
            sub_chunks = markdown_splitter(chunk.url, chunk.content, self.embedder.count_tokens, self.embedder.max_input_tokens)
            for sub_chunk in sub_chunks:
                # compute the embedding of the sub-chunk
                embedding = self.embedder.embed(sub_chunk.content)
                # add embedding to the vector database
                sub_text_index = self._index_add(embedding)
                # add index to file
                file.add_index(sub_text_index)
                # register chunk at the sub chunk's index
                self.chunks[sub_text_index] = chunk
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