"""
Used to store chunks, associaing them with a unique id
And keep track of files processed and when they were last processed
Does NOT deal with search
"""
import os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Any
from ..chunk import Chunk
from .file import File
from ..document_splitter import file_splitter

# set of file extension we drop
# run the following to check which extensions are currently in the doc: `find . -type f | awk -F. 'NF>1 {print $NF}' | sort | uniq`
# we are moslty interested in markdown, code, and script files
FORBIDDEN_EXTENSIONS = {'.gif', '.png', '.jpg', '.jpeg', '.css', '.gitkeep', '.pdf', '.in', '.out', '.output'}

# set of documenation folders we ignore
FORBIDDEN_FOLDERS = {'timeline'}

class DocumentStore:
    """
    Used to keep track of files and their chunks, associating an id with each chunk.
    """
    def __init__(self, database_folder:Path, documentation_folder:Path):
        """
        Takes the path where it should find the database as well as the path where it should find the documentation
        """
        self.documentation_folder = documentation_folder.absolute().resolve()
        # dictionary of all files
        self.files: Dict[Path, File] = dict() # file_path -> File(date,[chunk_id])
        # dictionary of all chunk
        self.chunks: Dict[int, Chunk] = dict() # chunk_id -> Chunk
        self.max_chunk_id = 0

    def get_chunk(self, id:int) -> Chunk:
        """
        Returns he chunk associaed wih a given id.
        Errors ou if it does no exist.
        """
        return self.chunks[id]

    #--------------------------------------------------------------------------
    # FILE PROCESSING

    def is_ignored_file(self, file_path:Path) -> bool:
        """Returns true if the file should be ignored"""
        ignored_folder = (file_path.name in FORBIDDEN_FOLDERS) or any(parent.name in FORBIDDEN_FOLDERS for parent in file_path.parents)
        ignored_extension = (file_path.suffix in FORBIDDEN_EXTENSIONS)
        return ignored_folder or ignored_extension

    def remove_file(self, file_path:Path) -> List[int]:
        """
        Removes a file's content from the DocumentStore.
        Returns a list of chunk indices to be deleted.
        """
        # load our data on the file
        file = self.files[file_path]
        chunk_indices_to_remove = file.chunk_indices
        # remove file chunks from chunks
        for index in chunk_indices_to_remove:
            del self.chunks[index]
        # remove file from files
        del self.files[file_path]
        return chunk_indices_to_remove

    def add_file(self, file_path:Path, token_counter: Callable[[str], int], max_tokens_per_chunk: int) -> dict[int,Chunk]:
        """
        Add a file's content to the DocumentStore.
        Returns a list of (chunk index,chunk) to be added.
        """
        # create a file for the DocumentStore
        file_update_date = datetime.fromtimestamp(file_path.stat().st_mtime)
        file = File(update_date=file_update_date)
        # slice text into chunks small enough for our needs
        chunks = file_splitter(self.documentation_folder, file_path, token_counter, max_tokens_per_chunk)
        new_chunks = {}
        for chunk in chunks:
            # gets an index for the chunk
            chunk_index = self.max_chunk_id
            self.max_chunk_id += 1
            # add chunk index to its file
            file.add_index(chunk_index)
            # add chunk to new chunks
            new_chunks[chunk_index] = chunk
        # add new chunks to chunks
        self.chunks.update(new_chunks)
        # add file to files
        self.files[file_path] = file
        return new_chunks

    def update(self, token_counter: Callable[[str], int], max_tokens_per_chunk: int, verbose=False) -> dict[str,Any]:
        """
        Scans the documentation folder, removes files that have been deleted or are outdated,
        and adds new files. It returns a dictionary with chunks that need to be added or removed.

        Args:
            token_counter (Callable[[str], int]): A function that returns the number of tokens in a given string.
            max_tokens_per_chunk (int): The maximum number of tokens allowed per chunk.
            verbose (bool, optional): If True, enables verbose output for debugging purposes. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'add_chunks' (dict): Chunks that need to be added.
                - 'remove_chunk_ids' (List[str]): IDs of chunks that need to be removed.
        """
        # list of chunk modifications, to be returned
        add_chunks = {}
        remove_chunk_ids = []
        # removes files that have been deleted or are out of date
        existing_files = list(self.files.items())
        for file_path, file in tqdm(existing_files, disable=not verbose, desc="Checking for old files"):
            if (not file_path.exists()) or (datetime.fromtimestamp(file_path.stat().st_mtime) > file.update_date):
                file_remove_chunk_ids = self.remove_file(file_path)
                remove_chunk_ids.extend(file_remove_chunk_ids)
        # gets relative paths for all documenaion files
        current_files = [Path(root).joinpath(file) for root, dirs, files in os.walk(self.documentation_folder) for file in files]
        # add new files
        if len(current_files) == 0:
            raise RuntimeError(f"ERROR: the documentation folder '{self.documentation_folder}' is empty or does not exist.")
        for file_path in tqdm(current_files, disable=not verbose, desc="Checking for new files"):
            if (not file_path in self.files) and (not self.is_ignored_file(file_path)):
                file_add_chunks = self.add_file(file_path, token_counter, max_tokens_per_chunk)
                add_chunks.update(file_add_chunks)
        # returns chunks that needs to be added / removed
        return {'add_chunks': add_chunks, 'remove_chunk_ids': remove_chunk_ids}

    #--------------------------------------------------------------------------
    # SERIALIZATION

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the class, for saving purposes"""
        return {
            'files': {str(k): v.to_dict() for k, v in self.files.items()},
            'chunks': {k: v.to_dict() for k, v in self.chunks.items()},
            'max_chunk_id': self.max_chunk_id
        }

    def from_dict(self, data: Dict[str, Any]):
        """Loads a dictionnary representation of the class"""
        self.files = {Path(k): File.from_dict(v) for k, v in data['files'].items()}
        self.chunks = {int(k): Chunk.from_dict(v) for k, v in data['chunks'].items()}
        self.max_chunk_id = int(data['max_chunk_id'])
