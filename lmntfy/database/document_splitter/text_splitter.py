import math
from typing import Callable, List
from ..chunk import Chunk

def text_splitter(url:str, text:str, token_counter:Callable[[str],int], max_tokens:int) -> List[Chunk]:
    """
    Splits a given text into chunks based on a maximum token limit.

    Args:
        url (srt): The URL where the tex can be found.
        text (str): The input text to be split.
        token_counter (Callable[[str], int]): A function that returns the number of tokens in a given string.
        max_tokens (int): The maximum number of tokens allowed in each chunk.

    Returns:
        List[Chunk]: A list of chunks, each having the given url and containing no more than the specified maximum number of tokens.

    The function processes the input text line by line, ensuring each chunk is just small enough to be valid.
    The splitting is done such that each token overlaps with the previous one by about a third.
    """
    # shortcut if the text is short enough to be returned uncut
    if token_counter(text) < max_tokens:
        return [Chunk(url, text)]
    # process line by line
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    for line in text.splitlines():
        line_size = token_counter(line)
        new_chunk_size = current_chunk_size + line_size
        if new_chunk_size < max_tokens:
            # adds the line to the chunk
            current_chunk.append(line)
            current_chunk_size = new_chunk_size
        else:
            # the chunk will get too large, start a new one
            if len(current_chunk) > 0: 
                # saves the chunk
                current_chunk_text = '\n'.join(current_chunk)
                chunks.append(current_chunk_text)
            # starts a new chunk, overlapping the previous one by a third
            index_one_third = int(math.ceil(len(current_chunk) / 3))
            current_chunk = current_chunk[index_one_third:] if (index_one_third < len(current_chunk)) else []
            current_chunk_size = sum((token_counter(line) for line in current_chunk), 0)
    # turn leftover lines into a last chunk
    if len(current_chunk) > 0: 
        current_chunk_text = '\n'.join(current_chunk)
        chunks.append(current_chunk_text)
    # return the chunks produced
    return [Chunk(url, content) for content in chunks]
