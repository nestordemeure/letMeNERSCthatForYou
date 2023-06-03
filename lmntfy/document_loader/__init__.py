import os
from ..models.llm import LanguageModel
from .text_spliter import text_splitter
from .markdown_spliter import markdown_splitter


class DocumentLoader:
    """
    Takes a folder or documents
    turns them into chunks
    """
    def __init__(self, llm: LanguageModel, target_chunk_per_query=3):
        # the model we will use
        self.model = llm
        # the resulting chunks
        self.chunks = []
        # maximum size of each chunk
        # we leave space for two additional chunks, representing the prompt and the model's answer
        self.max_chunk_size = self.model.context_size / (target_chunk_per_query + 2)
        # the token counting function
        self.token_counter = self.model.token_counter

    def load_folder(self, folder_path, verbose=False):
        """Adds all markdown documents in a folder to the Splitter."""
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                self.load_file(file_path, verbose=verbose)

    def load_file(self, file_path, verbose=False):
        """Adds a markdown document to the Splitter, splitting it until it fits."""
        try:
            with open(file_path, 'r', encoding='utf8') as file:
                text = file.read()
                # split the file into chunks
                if file_path.endswith('.md'): 
                    raw_chunks = markdown_splitter(text, self.token_counter, self.max_chunk_size)
                else:
                    raw_chunks = text_splitter(text, self.token_counter, self.max_chunk_size)
                # saves all the chunks
                for content in raw_chunks:
                    self.load_chunk(file_path, content)
                print(f"Loaded file '{file_path}' ({len(raw_chunks)} chunks)")
        except UnicodeDecodeError:
            # unsupported format
            print(f"WARNING: File '{file_path}' does not appear to be a utf8 encoded text file.")

    def load_chunk(self, document_path, text):
        """Adds a given text and source to the Splitter"""
        chunk = {'source':document_path, 'content':text.strip()}
        self.chunks.append(chunk)
    
    def documents(self):
        """Returns a list of all the chunks produced so far."""
        return self.chunks
