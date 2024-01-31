import re
from pathlib import Path
from .text_spliter import text_splitter
from .markdown_spliter import markdown_splitter
from .chunk import Chunk, path2url
from .token_count_pair import TokenCountPair
from typing import Callable, List

def paths2urls(markdown, file_path, folder_path):
    """
    Takes a markdown and turns all of its relative paths into urls
    """
    # turns the relative path into a proper url
    def replacer(match):
        # gets raw markdown link information
        link_name = match.group(1)
        link_relative_path = Path(match.group(2))
        # turn the relative path (computed from the containing folder's perspective) into a url
        while True:
            try:
                # gets an absolute path
                link_path = (file_path.parent / link_relative_path).resolve()
                # turns it into a url
                link_url = path2url(link_path, folder_path)
                return '[{}]({})'.format(link_name, link_url)
            except:
                # remove the first part of the relative path (usually one `../` too many)
                parts = link_relative_path.parts[1:]
                link_relative_path = Path(*parts)
    # matches markdown links patterns where the link does not start with http
    # (hinting at the fact that it is a relative path)
    return re.sub(r'\[([^]]+)\]\(((?!http)[^)]+)\)', replacer, markdown)

def chunk_file(file_path:Path, documentation_folder:Path, token_counter:Callable[[str],TokenCountPair], max_tokens_per_chunk: TokenCountPair, verbose=False) -> List[Chunk]:
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

