import json
from pathlib import Path
from typing import Callable, List, Dict, Set
from ..models import LanguageModel
from .chunk import Chunk
from .document_store import DocumentStore
from .search import SearchEngine

class Database:
    """Vector Database"""
    def __init__(self, documentation_folder:Path, database_folder:Path,
                       search_engine:SearchEngine, llm: LanguageModel, 
                       min_chunks_per_query=8, max_tokens_per_chunk=None, update_database=True):
        """
        Initialize the Database instance.

        Parameters:
            documentation_folder (Path): The path to the folder containing the documentation.
            database_folder (Path): The path to the folder where the database will be stored.
            search_engine (SearchEngine): The search engine to be used for querying the database.
            llm (LanguageModel): The language model that will receive the chunks.
            min_chunks_per_query (int, optional): The target minimum number of chunks per query. Default is 8.
            max_tokens_per_chunk (int, optional): The maximum number of tokens per chunk. If None, it is calculated based on the language model's context size and min_chunks_per_query.
            update_database (bool, optional): Whether to update the database to the latest documentation. Default is True.
        """
        # parameters
        self.max_tokens_per_chunk = max_tokens_per_chunk if (max_tokens_per_chunk is not None) else int(llm.context_size / (min_chunks_per_query + 2))
        # names and paths
        self.name = f"{llm.tokenizer.name}{self.max_tokens_per_chunk}_{search_engine.name}"
        self.documentation_folder = documentation_folder.absolute().resolve()
        self.database_folder = database_folder.absolute().resolve() / self.name
        # components
        self.search_engine = search_engine
        self.search_engine.initialize(self.database_folder)
        self.document_store = DocumentStore(self.database_folder, self.documentation_folder)
        # loads the database from file if possible
        if self.exists():
            self.load()
        # updates the database to the latest documentation
        if update_database or not self.exists():
            self.update(llm.count_tokens, self.max_tokens_per_chunk, verbose=True)
    
    def get_closest_chunks(self, input_text: str, k: int = 3, verbose=False) -> List[Chunk]:
        """
        Returns the (at least) k chunks that are relevant according to he search engine
        """
        # shortcut if we ask for more chunks than available
        if len(self.document_store.chunks) <= k:
            return list(self.document_store.chunks.values())
        # queries the search engine
        scored_chunk_id = self.search_engine.get_closest_chunks(input_text, self.document_store.chunks, k)
        # gets he chunks from the document store
        chunks = [self.document_store.get_chunk(id) for (score,id) in scored_chunk_id]
        # debug information
        if verbose:
            print(f"\nQ: {input_text}")
            for i in range(len(chunks)):
                print(f" * [{scored_chunk_id[i][0]:.2f}]: {chunks[i].url}")
        # returns
        return chunks

    def update(self, token_counter: Callable[[str], int], max_tokens_per_chunk: int, verbose=False):
        """Goes over the documentation and insures that we are up to date then saves the result."""
        # chunk current files
        delta_chunks = self.document_store.update(token_counter, max_tokens_per_chunk, verbose=verbose)
        # add / remove chunks from the search engine
        self.search_engine.remove_several_chunks(delta_chunks['remove_chunk_ids'])
        self.search_engine.add_several_chunks(delta_chunks['add_chunks'])
        # save resulting database
        self.save()

    def exists(self):
        """
        Returns True if the database already exists on disk.
        """
        document_store_path = self.database_folder / 'document_store.json'
        return self.database_folder.exists() and document_store_path.exists() and self.search_engine.exists(self.database_folder)

    def load(self):
        # load the document store
        with open(self.database_folder / 'document_store.json', 'r') as f:
            document_store_dic = json.load(f)
            self.document_store.from_dict(document_store_dic)
        # load the search engine
        self.search_engine.load(self.database_folder)

    def save(self):
        # insures that the saving folder exists
        self.database_folder.mkdir(parents=True, exist_ok=True)
        # saves the document store
        with open(self.database_folder / 'document_store.json', 'w') as f:
            document_store_dic = self.document_store.to_dict()
            json.dump(document_store_dic, f)
        # saves the search engine
        self.search_engine.save(self.database_folder)
