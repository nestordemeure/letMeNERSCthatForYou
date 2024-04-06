import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from abc import ABC, abstractmethod
from ..models import LanguageModel, Embedding, embedding, Reranker, reranker
from .document_loader import Chunk, chunk_file
from .document_loader.markdown_spliter import markdown_splitter
from .utilities.file import File

def remove_duplicates(chunks:List[Chunk]) -> Chunk:
    """remove duplicates from a list while preserving element order"""
    seen = set()
    result = []
    for item in chunks:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

class Database(ABC):
    """Vector Database"""
    def __init__(self, documentation_folder:Path, database_folder:Path,
                       llm: LanguageModel, embedder: Embedding, reranker: Reranker,
                       min_chunks_per_query=8, update_database=True, name:str=''):
        # names and paths
        self.name = name
        self.documentation_folder = documentation_folder.absolute().resolve()
        self.database_folder = database_folder.absolute().resolve() / (self.name + '_' + embedder.name + '_' + reranker.name  + '_' + llm.tokenizer.name)
        # models
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        # parameters
        self.min_chunks_per_query = min_chunks_per_query
        # NOTE: we leave space for two additional chunks, representing the prompt and the model's answer
        # dictionary of all files
        self.files: Dict[Path, File] = dict() # file_path -> File
        # dictionary of all chunk
        # NOTE: it might contain identical chunks pointed at by different indices
        self.chunks: Dict[int, Chunk] = dict() # vector_database_index -> Chunk
        # set of file and folder names that will not be processed
        self.ignored_files_and_folders: Set[str] = {'timeline'}
        # loads the database from file if possible
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
        # we might want to query more chunks than needed then keep the best ones
        use_reranker = not isinstance(self.reranker, reranker.NoReranker)
        nb_chunks_needed = (2*k) if use_reranker else k
        # gets the chunks
        if len(self.chunks) <= nb_chunks_needed:
            # we get all the chunks we have
            chunks = list(self.chunks.values())
        else:
            # assumes the input is small enough to be embedded
            input_embedding = self.embedder.embed(input_text, is_query=True)
            # Loop until we get enough *distinct* chunks
            # (duplicates can stem from embedding the chunk as several subtexts to fit in the embedder context size)
            chunks = list()
            growing_factor = 1
            while len(chunks) < nb_chunks_needed:
                # query the vector database
                chunks_indices = self._index_get_closest(input_embedding, nb_chunks_needed*growing_factor)
                # ensures uniqueness
                chunks = remove_duplicates([self.chunks[i] for i in chunks_indices])
                # the next call will request twice as many items
                growing_factor *= 2
        # reranks the chunks
        chunks = self.reranker.rerank(input_text, chunks) if use_reranker else chunks
        # keep only the required number of chunks
        return chunks[:k] if (len(chunks) > k) else chunks

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
        max_tokens_per_chunk = self.llm.context_size / (self.min_chunks_per_query + 2)
        chunks = chunk_file(file_path, self.documentation_folder, self.llm.count_tokens, max_tokens_per_chunk)
        for chunk in chunks:
            # slice chunk into sub chunks small enough to be embedded
            sub_chunks = markdown_splitter(chunk.url, chunk.content, self.embedder.count_tokens, self.embedder.context_size)
            for sub_chunk in sub_chunks:
                # compute the embedding of the sub-chunk
                embedding = self.embedder.embed(sub_chunk.content)
                # add embedding to the vector database
                sub_text_index = self._index_add(embedding)
                # add index to file
                file.add_index(sub_text_index)
                # register chunk at the sub chunk's index
                self.chunks[sub_text_index] = chunk
            # NOTE: we have embeddings pointing at parts of the full chunk
            # averaging them to get a chunk's embedding also somewhat works
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
from .whoosh import WhooshDatabase
# the database used by default everywhere
Default = FaissDatabase