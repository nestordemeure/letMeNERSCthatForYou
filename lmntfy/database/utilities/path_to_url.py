"""
Utilities to convert paths and headers into urls
"""
from pathlib import Path
from urllib.parse import quote
import re

def path2url(file_path:Path) -> str:
    """
    Take a file path inside the NERSC documentation and turns it into a url.
    """
    # Construct the new URL
    url = "https://docs.nersc.gov/" + str(file_path)
    # Replace an optional '/index.md' with '/'
    url = url.replace("/index.md", "/")
    # Replace the final '.md' with '/'
    url = url.replace(".md", "/")
    # Convert spaces into URL valid format
    url = quote(url, safe='/:#')
    return url

def resolve_all_paths2urls(markdown:str, file_path:Path) -> str:
    """
    Takes a markdown and turns all of its relative paths into urls.
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
                link_url = path2url(link_path)
                return '[{}]({})'.format(link_name, link_url)
            except:
                # remove the first part of the relative path (usually one `../` too many)
                parts = link_relative_path.parts[1:]
                link_relative_path = Path(*parts)
    # matches markdown links patterns where the link does not start with http
    # (hinting at the fact that it is a relative path)
    return re.sub(r'\[([^]]+)\]\(((?!http)[^)]+)\)', replacer, markdown)

def addHeader2url(url:str, header:str) -> str:
    """
    Takes a url and the header of a markdown section inside that file,
    converts it into a url to the relevant part of that page.
    """
    # extract the heading
    lines = header.splitlines()
    heading = lines[0] if header.startswith('#') else ""
    # converts the heading into a proper url format    
    # Convert to lowercase
    heading = heading.lower()
    # Remove all non-alphanumeric characters
    heading = re.sub('[^a-z0-9- ]', '', heading)
    # strip side spaces
    heading = heading.strip()
    # Replace spaces with dashes
    heading = heading.replace(' ', '-')
    # Add back the hash sign at the start for URL fragment identifier
    heading = '#' + heading
    # merges it with the url
    return url + heading