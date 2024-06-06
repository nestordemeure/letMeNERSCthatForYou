"""
Utilities to split various document types into properly sized chunks
"""
from pathlib import Path
from typing import Callable, List
from ..chunk import Chunk
from .path_to_url import path2url, resolve_all_paths2urls
from .text_splitter import text_splitter
from .markdown_splitter import markdown_splitter

def file_splitter(documentation_folder:Path, file_path: Path, token_counter: Callable[[str], int], max_tokens_per_chunk: int) -> List[Chunk]:
    """
    Splits a file into chunks based on a maximum token limit.

    Args:
        documentation_folder (Path): The path to the folder that contains the documentation, used to build relative paths.
        file_path (Path): The path to the file that needs to be split.
        token_counter (Callable[[str], int]): A function that returns the number of tokens in a given string.
        max_tokens_per_chunk (int): The maximum number of tokens allowed in each chunk.

    Returns:
        List[Chunk]: A list of chunks, each containing no more than the specified maximum number of tokens.

    NOTE:
    this function will turn relative paths inside the text into proper urls
    and might return an empty list if the file is invalid
    """
    # try loading the file
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            # turns the path into a relative path
            file_path = file_path.relative_to(documentation_folder)
            # reads the text ensuring that all paths are proper urls
            text = file.read()
            text = resolve_all_paths2urls(text, file_path)
            # plit it into chuncks
            chunks = list()
            url = path2url(file_path)
            if file_path.suffix == '.md': 
                # splits the text along headings when possible
                chunks = markdown_splitter(url, text, token_counter, max_tokens_per_chunk)
            else:
                # gets raw pieces of text
                chunks = text_splitter(url, text, token_counter, max_tokens_per_chunk)
            # returns only non-empty chunks
            return [chunk for chunk in chunks if len(chunk.content) > 0]
    except UnicodeDecodeError:
        # unsupported format
        print(f"WARNING: File '{file_path}' does not appear to be a utf8 encoded text file.")
        return list()

def chunk_splitter(chunk: Chunk, token_counter: Callable[[str], int], max_tokens_per_chunk: int) -> List[Chunk]:
    """
    Splits a chunk into sub-chunks.

    Args:
        chunk (Chunk): The chunk that needs to be split.
        token_counter (Callable[[str], int]): A function that returns the number of tokens in a given string.
        max_tokens_per_chunk (int): The maximum number of tokens allowed in each chunk.

    Returns:
        List[Chunk]: A list of chunks, each containing no more than the specified maximum number of tokens.
    """
    if chunk.is_markdown:
        return markdown_splitter(chunk.url, chunk.content, token_counter, max_tokens_per_chunk)
    else:
        return text_splitter(chunk.url, chunk.content, token_counter, max_tokens_per_chunk)
