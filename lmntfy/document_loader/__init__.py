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

    def load_folder(self, folder_path):
        """Adds all markdown documents in a folder to the Splitter."""
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                self.load_file(file_path)

    def load_file(self, file_path):
        """Adds a markdown document to the Splitter, splitting it until it fits."""
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read()
            # split the file into chunks
            if file_path.endswith('.md'): 
                raw_chunks = markdown_splitter(text, self.token_counter, self.max_chunk_size)
            elif file_path.endswith('.txt'): 
                raw_chunks = text_splitter(text, self.token_counter, self.max_chunk_size)
            else:
                # unsupported format
                # TODO add a verbose print statement
                return
            # saves all the chunks
            for content in raw_chunks:
                self.load_chunk(file_path, content)

    def load_chunk(self, document_path, text):
        """Adds a given text and source to the Splitter"""
        chunk = {'source':document_path, 'content':text.strip()}
        self.chunks.append(chunk)
    
    def documents(self):
        """Returns a list of all the chunks produced so far."""
        return self.chunks
