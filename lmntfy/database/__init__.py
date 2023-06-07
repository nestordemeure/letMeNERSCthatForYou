from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from ..models.embedding import Embedding
from concurrent.futures import ThreadPoolExecutor

class Database(ABC):
    def __init__(self, embedder:Embedding):
        self.embedder = embedder

    def add_chunks(self, chunks, verbose=False):
        """Adds several documents to the database."""
        for chunk in tqdm(chunks, disable=not verbose, desc="Loading chunks"):
            self.add_chunk(chunk)

    def concurrent_add_chunks(self, chunks, verbose=False):
        """
        Adds several documents to the database.
        This version is faster than `add_chunks` but might trigger an overusage error from OpenAI.
        """
        # using `with` ensures that we wait for all addition to complete
        with ThreadPoolExecutor() as executor:
            tasks = list(executor.map(self.add_chunk, chunks))
            # optional progress bar, tracking the progress of the tasks
            if verbose:
                for _ in tqdm(tasks, total=len(chunks), desc='Adding chunks'):
                    pass

    @abstractmethod
    def add_chunk(self, chunk):
        """
        Abstract method for adding a new document to the database.
        """
        pass

    @abstractmethod
    def get_closest_chunks(self, input_text:str, k=3):
        """
        Abstract method for finding the k closest documents to the input_text in the database.
        """
        pass

    @abstractmethod
    def save_to_file(self, file_path:str):
        pass

    @abstractmethod
    def load_from_file(self, file_path:str):
        """
        Abstract method for loading the database from a file.
        """
        pass

from .naive import NaiveDatabase
from .faiss import FaissDatabase