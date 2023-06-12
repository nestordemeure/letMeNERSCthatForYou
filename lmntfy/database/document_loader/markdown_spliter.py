import re
from .text_spliter import text_splitter
from .token_count_pair import TokenCountPair
from typing import Callable, List

#----------------------------------------------------------------------------------------
# MARKDOWN PARSING

class Markdown:
    def __init__(self, header: str, level: int, headings: list=[]):
        self.header = header 
        self.level = level
        self.headings = headings
        self.nb_tokens = None

    @staticmethod
    def load(markdown_text: str):
        result = Markdown(header="", level=0, headings=[])
        in_code_block = False
        for line in markdown_text.splitlines():
            # flip the state if this is a codeblock
            if line.startswith("```"):
                in_code_block = not in_code_block
            # inserts either text or a heading
            match = re.match(r'^#+', line)
            if match and not in_code_block:
                level = len(match.group())
                result.insert_heading(text=line, level=level)
            else:
                result.insert_text(line)
        # pop the upper level if it is empty
        if (len(result.header) == 0) and (len(result.headings) == 1):
            result = result.headings[0]
        return result
    
    def insert_text(self, text):
        """insert text at the end of the header of the latest, deepest, heading to data"""
        if len(self.headings) == 0:
            # we have reached the bottom
            self.header += '\n' + text
        else:
            self.headings[-1].insert_text(text)

    def insert_heading(self, text, level):
        """insert a new heading whereaver is possible"""
        if (len(self.headings) == 0) or (level <= self.headings[-1].level):
            heading = Markdown(text, level, headings=[])
            self.headings.append(heading)
        else:
            self.headings[-1].insert_heading(text, level)

    def to_string(self):
        result = self.header + '\n'
        for heading in self.headings:
            result += '\n' + heading.to_string()
        return result

    def count_tokens(self, token_counter):
        """memoized token counting function"""
        if self.nb_tokens is None:
            self.nb_tokens = token_counter(self.header) + sum((heading.count_tokens(token_counter) for heading in self.headings), TokenCountPair(0,0))
        return self.nb_tokens

    def to_chunks(self, token_counter, max_tokens):
        if (self.count_tokens(token_counter) < max_tokens) and (len(self.headings) > 0):
            # small enough to fit
            return [self.to_string()]
        else:
            # split along headings
            # include header only if it has more than one line
            header = self.header.strip()
            result = text_splitter(header, token_counter, max_tokens) if ('\n' in header) else []
            # include all subheadings
            for heading in self.headings:
                result.extend(heading.to_chunks(token_counter, max_tokens))
            return result

def markdown_splitter(markdown: str, token_counter:Callable[[str],TokenCountPair], max_tokens:TokenCountPair) -> List[str]:
    """
    takes a markdown file as a string
    a function that can count the number of tokens in a string
    and a maximum number of tokens to be accepted

    parses the markdown into a tree representation along its headings
    cut down recurcively until it is small enough to fit out maximum number of tokens allowed
    """
    # shortcut if the text is short enough to be returned uncut
    if token_counter(markdown) < max_tokens:
        return [markdown]
    # parses the text into a tree representation
    ast = Markdown.load(markdown)
    # turn it into a list of chunks of appropriate size
    return ast.to_chunks(token_counter, max_tokens)
