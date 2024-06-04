"""
Utilities to split various document types into properly sized chunks
"""
from pathlib import Path
from .text_splitter import text_splitter
from .markdown_splitter import markdown_splitter
from ..utilities.chunk import Chunk, path2url
from typing import Callable, List

def chunk_file(file_path:Path, documentation_folder:Path, token_counter:Callable[[str],int], max_tokens_per_chunk: int, verbose=False) -> List[Chunk]:
    """Adds a markdown document to the Splitter, splitting it until it fits."""
    # generates the url to the file
    url = path2url(file_path, documentation_folder)
    # assembles the chunks
    result = list()
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read()
            # insures all links inside the file are urls
            text = paths2urls(text, file_path, documentation_folder)
            # split the file into chunks
            if file_path.suffix == '.md': 
                # splits the text along headings when possible
                chunks = markdown_splitter(url, text, token_counter, max_tokens_per_chunk)
            else:
                # gets raw pieces of text
                raw_chunks = text_splitter(text, token_counter, max_tokens_per_chunk)
                chunks = [Chunk(url=url, content=content) for content in raw_chunks]
            # saves the non-empty chunks
            for chunk in chunks:
                if len(chunk.content) > 0:
                    # saves the chunk
                    result.append(chunk)
            if verbose: print(f"Loaded file '{file_path}' ({len(result)} chunks)")
    except UnicodeDecodeError:
        # unsupported format
        if verbose: print(f"WARNING: File '{file_path}' does not appear to be a utf8 encoded text file.")
    return result

