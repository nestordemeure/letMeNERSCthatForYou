import re
from typing import List
from ..database.document_splitter import Chunk

# references to be returned in the absence of further information
DEFAULT_REFERENCES = ["https://docs.nersc.gov/", "https://www.nersc.gov/users/getting-help/online-help-desk/"]

def stem_url(url: str) -> str:
    """
    Takes a URL and cuts it at the latest '#' if possible.
    
    Args:
        url (str): The URL to be processed.
        
    Returns:
        str: The stemmed URL without the fragment part after the latest '#'.
    """
    # Find the position of the last occurrence of '#'
    hash_position = url.rfind('#')
    # If '#' is found, return the URL up to (but not including) the '#'
    if hash_position != -1:
        return url[:hash_position]
    # If '#' is not found, return the original URL
    return url

def validate_references(urls:List[str], chunks:List[Chunk], prompt:str, stem_before_validation=False) -> List[str]:
    """
    Takes:
    - urls: a list of urls to validate
    - chunks: a list of chnuks used to build the answer
    - prompt: the prompt which contains the conversation so far
    - stem_before_validation (bool): should we ignore the # section at the end of the url when validating?

    Returns urls that are:
    - a chunk's url OR previously referenced OR in a chunk

    Returns DEFAULT_REFERENCES if no valid reference is found.
    """
    # the stemmer function
    stemmer = stem_url if stem_before_validation else (lambda url: url)

    # urls of the chunks
    chunk_urls = {stemmer(chunk.url) for chunk in chunks}
    # urls referenced in the conversation so far
    reference_pattern = r"\* \<([^\>]*?)\>"
    prompt_urls = {stemmer(url) for url in re.findall(reference_pattern, prompt)}
    # set of urls accepted
    accepted_urls = chunk_urls | prompt_urls

    # keep only urls that are previously referenced OR a chunk url OR in a chunk
    valid_urls = set()
    for url in urls:
        stemmed_url = stemmer(url)
        if ((stemmed_url in accepted_urls) or any((stemmed_url in chunk.content) for chunk in chunks)):
            # known url
            valid_urls.add(url)
        # NOTE: experiemnt with opening us up to generated urls
        #elif (not url.startswith('https://docs.nersc.gov/')) and validators.url(url):
        #    # valid, unknown, url
        #    url = f"{url} (VALIDATED)"
        #    valid_urls.add(url)
    
    # returns, providing default urls if none was valid
    return DEFAULT_REFERENCES if (len(valid_urls) == 0) else valid_urls

def format_reference_list(urls: List[str]) -> str:
    """
    takes a list of urls
    produce a single string representing them as a bullet list of markdown urls (` * <url>`)
    """
    formatted_list = ""
    for url in urls:
        # appends the url, formated as a markdown url
        formatted_list += f" * <{url}>\n"
    # returns, removing the trailing newline
    return formatted_list.rstrip()