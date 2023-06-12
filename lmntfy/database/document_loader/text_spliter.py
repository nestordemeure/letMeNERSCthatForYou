import math
from .token_count_pair import TokenCountPair
from typing import Callable, List

def text_splitter(text: str, token_counter:Callable[[str],TokenCountPair], max_tokens:TokenCountPair) -> List[str]:
    """
    takes a text as a string
    a function that can count the number of tokens in a string
    and a maximum number of tokens to be accepted

    the splitting is done line by line
    each chunk is just enough lines to be valid (adding one line would make it invalid)
    we do the splitting by thirds, meaning that each third of a chunk is the center of its own section
    """
    # shortcut if the text is short enough to be returned uncut
    if token_counter(text) < max_tokens:
        return [text]
    # cut text into lines
    lines = text.splitlines()
    # process line by line
    chunks = []
    current_chunk = []
    current_chunk_size = TokenCountPair(0,0)
    for line in lines:
        line_size = token_counter(line)
        new_chunk_size = current_chunk_size + line_size
        if new_chunk_size < max_tokens:
            current_chunk.append(line)
            current_chunk_size = new_chunk_size
        else:
            if len(current_chunk) > 0: 
                current_chunk_text = '\n'.join(current_chunk)
                chunks.append(current_chunk_text)
            index_one_third = int(math.ceil(len(current_chunk) / 3))
            current_chunk = current_chunk[index_one_third:] if (index_one_third < len(current_chunk)) else []
            current_chunk_size = sum((token_counter(line) for line in current_chunk), TokenCountPair(0,0))
    # add leftover lines
    if len(current_chunk) > 0: 
        current_chunk_text = '\n'.join(current_chunk)
        chunks.append(current_chunk_text)
    # return
    return chunks
