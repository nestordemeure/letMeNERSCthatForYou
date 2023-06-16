from pathlib import Path
from .text_spliter import text_splitter
from .markdown_spliter import markdown_splitter
from .chunk import Chunk, path2url
from .token_count_pair import TokenCountPair
from typing import Callable, List

def chunk_file(file_path:Path, url:str, token_counter:Callable[[str],TokenCountPair], max_tokens_per_chunk: TokenCountPair, verbose=False) -> List[Chunk]:
    """Adds a markdown document to the Splitter, splitting it until it fits."""
    result = list()
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read()
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
                    result.append(chunk)
            if verbose: print(f"Loaded file '{file_path}' ({len(result)} chunks)")
    except UnicodeDecodeError:
        # unsupported format
        if verbose: print(f"WARNING: File '{file_path}' does not appear to be a utf8 encoded text file.")
    return result

