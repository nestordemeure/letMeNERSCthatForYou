"""
Utilities to split various document types into properly sized chunks
"""
from pathlib import Path
from typing import Callable, List
from ..utilities import Chunk
from .path_to_url import path2url, resolve_all_paths2urls
from .text_splitter import text_splitter
from .markdown_splitter import markdown_splitter

# list of file extension we drop
# run the following to check which extensions are currently in he doc: `find . -type f | awk -F. 'NF>1 {print $NF}' | sort | uniq`
# we are moslty interested in markdown, code, and script files
FORBIDDEN_EXTENSIONS = ['gif', 'png', 'jpg', 'jpeg', 'css', 'gikeep', 'pdf', 'in', 'out', 'output']

def file_splitter(file_path: Path, token_counter: Callable[[str], int], max_tokens_per_chunk: int, verbose=False) -> List[Chunk]:
    """
    Splits a file into chunks based on a maximum token limit and saves the chunks in a specified documentation folder.

    Args:
        file_path (Path): The path to the file that needs to be split.
        token_counter (Callable[[str], int]): A function that returns the number of tokens in a given string.
        max_tokens_per_chunk (int): The maximum number of tokens allowed in each chunk.
        verbose (bool, optional): If True, enables verbose output for debugging purposes. Defaults to False.

    Returns:
        List[Chunk]: A list of chunks, each containing no more than the specified maximum number of tokens.

    NOTE:
    * this function will turn relative paths inside the text into proper urls
    * it will also skip any file that is not considered a documenation file (ie: images)
    """
    # shortcut for unsupported file formats
    if file_path.suffix in FORBIDDEN_EXTENSIONS:
        # unsupported format
        return list()
    # try loading the file
    try:
        with open(file_path, 'r', encoding='utf8') as file:
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
            # keep only non-empty chunks
            chunks = [chunk for chunk in chunks if len(chunk.content) > 0]
            # returns
            if verbose: print(f"Loaded file '{file_path}' ({len(chunks)} chunks)")
            return chunks
    except UnicodeDecodeError:
        # unsupported format
        if verbose: print(f"WARNING: File '{file_path}' does not appear to be a utf8 encoded text file.")
        return list()
