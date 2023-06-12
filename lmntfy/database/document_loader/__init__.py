from pathlib import Path
from .text_spliter import text_splitter
from .markdown_spliter import markdown_splitter
from .chunk import Chunk
from .token_count_pair import TokenCountPair
from typing import Callable, List

def chunk_file(file_path:Path, token_counter:Callable[[str],TokenCountPair], max_tokens_per_chunk: TokenCountPair, verbose=False) -> List[Chunk]:
    """Adds a markdown document to the Splitter, splitting it until it fits."""
    chunks = list()
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read()
            # split the file into chunks
            if file_path.suffix == '.md': 
                raw_chunks = markdown_splitter(text, token_counter, max_tokens_per_chunk)
            else:
                raw_chunks = text_splitter(text, token_counter, max_tokens_per_chunk)
            # saves all the chunks
            for content in raw_chunks:
                if len(content) > 0:
                    chunk = Chunk(source=file_path, content=content.strip())
                    chunks.append(chunk)
            if verbose: print(f"Loaded file '{file_path}' ({len(raw_chunks)} chunks)")
    except UnicodeDecodeError:
        # unsupported format
        if verbose: print(f"WARNING: File '{file_path}' does not appear to be a utf8 encoded text file.")
    return chunks

