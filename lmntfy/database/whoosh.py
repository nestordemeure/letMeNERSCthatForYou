import json
from pathlib import Path
from typing import List, Dict, Set
from .document_loader import Chunk, chunk_file
from . import Database
from ..models import LanguageModel, Embedding
from .file import File
from datetime import datetime
from whoosh.index import exists_in, create_in, open_dir
from whoosh.fields import Schema, ID, TEXT
from whoosh.query import Term, Phrase
from whoosh.qparser import QueryParser
from whoosh import qparser

# Define the schema for our Chunk index
CHUNK_SCHEMA = Schema(
    filepath=TEXT(stored=True),
    url=TEXT(stored=True),
    content=TEXT(stored=True)
)

class WhooshDatabase(Database):
    """
    Traditional search algorithm instead of a vector database.
    WARNING: as the index update requires modifications on file, it cannot be updated from a compute node.
    """
    def __init__(self, llm:LanguageModel, embedder:Embedding,
                       documentation_folder:Path, database_folder:Path, 
                       min_chunks_per_query=8, update_database=True, name='whoosh'):
        # whoosh database that will be used to store the chunks
        self.index = None
        self.current_id = 0
        # conclude the initialisation
        super().__init__(llm, embedder, documentation_folder, database_folder, min_chunks_per_query, update_database, name)

    def _index_add(self, embedding) -> int:
        """
        Abstract method for adding a new vector to the vector database
        returns its index
        """
        raise RuntimeError("This method should never be called.")

    def _index_remove_several(self, indices: List[int]):
        """
        Abstract method for removing vectors from the vector database
        """
        raise RuntimeError("This method should never be called.")

    def _index_get_closest(self, input_embedding, k=3) -> List[int]:
        """
        Abstract method, returns the indices of the k closest embeddings in the vector database
        """
        raise RuntimeError("This method should never be called.")

    def get_closest_chunks(self, input_text: str, k: int = 3) -> List[Chunk]:
        """
        returns the (at most) k chunks that contains pieces of text closest to the input_text
        """
        if len(self.chunks) <= k: return list(self.chunks.values())
        # does a search in the index
        chunks = []
        with self.index.searcher() as searcher:
            # match all documents that contains at least one of the terms
            # TODO add some fuzzyness?
            query = QueryParser("content", schema=self.index.schema, group=qparser.OrGroup).parse(input_text)
            results = searcher.search(query, limit=k)
            print(f"DEBUGGING: keywords:{input_text}")
            for hit in results:
                print(f"DEBUGGING: url:{hit['url']} score:{hit.score}")
                # relevancy = hit.score
                chunk = Chunk(url=hit['url'], content=hit['content'])
                chunks.append(chunk)
        return chunks

    def remove_file(self, file_path:Path):
        """Removes a file's content from the Database"""
        file = self.files[file_path]
        indices_to_remove = file.vector_database_indices
        # remove file chunks from chunks
        for index in indices_to_remove:
            del self.chunks[index]
        # remove file's chunks from index
        # TODO delete_by_term(fieldname, termtext)
        writer = self.index.writer()
        with self.index.searcher() as searcher:
            # Use Term to search for the document by filepath
            query = Term("filepath", str(file_path))
            results = searcher.search(query, limit=None)
            for hit in results:
                writer.delete_document(hit.docnum)
        writer.commit()
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
        writer = self.index.writer()
        for chunk in chunks:
            # get current chunk index
            chunk_index = self.current_id
            # add chunk to the database
            writer.add_document(
                filepath=str(file_path),
                url=chunk.url,
                content=chunk.content)
            self.current_id += 1
            # add index to file
            file.add_index(chunk_index)
            # register chunk at the sub chunk's index
            self.chunks[chunk_index] = chunk
        writer.commit()
        # add file to files
        self.files[file_path] = file

    def exists(self):
        """returns True if the database already exists on disk"""
        database_data_file = self.database_folder / 'whoosh_database.json'
        return self.database_folder.exists() and database_data_file.exists() and exists_in(self.database_folder)

    def save(self):
        """ensures the database is saved"""
        # first save the rest of the database, ensuring the folder exists
        super().save()
        # saves the database data
        with open(self.database_folder / 'whoosh_database.json', 'w') as f:
            database_data = {'current_id': self.current_id}
            json.dump(database_data, f)
        # NOTE: we don't save the index as `writer.commit` already takes care of it

    def load(self, update_database=True, verbose=False):
        """loads the database"""
        if self.exists():
            # load the index
            self.index = open_dir(self.database_folder)
            # load the database_data
            with open(self.database_folder / 'whoosh_database.json', 'r') as f:
                database_data = json.load(f)
                self.current_id = int(database_data['current_id'])
        else:
            # creates a new index
            self.index = create_in(self.database_folder, CHUNK_SCHEMA)
            self.current_id = 0
        # loads the rest of the database and optionaly updates it
        super().load(update_database=update_database, verbose=verbose)
