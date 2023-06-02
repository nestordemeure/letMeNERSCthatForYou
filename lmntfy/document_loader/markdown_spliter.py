import mistune
from .text_spliter import text_splitter

#----------------------------------------------------------------------------------------
# MARKDOWN PARSING

class MarkdownRenderer(mistune.AstRenderer):
    """mistune makdown renderer to guide the parsing"""
    def __init__(self, token_counter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_counter = token_counter
        self.current_section = None
        self.sections = []

    def header(self, text, level):
        if self.current_section is not None:
            self.sections.append(self.current_section)
        self.current_section = Markdown(text, [], self.token_counter)

    def text(self, text):
        if self.current_section is not None:
            self.current_section.header += '\n' + text.strip()

    def finish(self):
        if self.current_section is not None:
            self.sections.append(self.current_section)
        return self.sections

#----------------------------------------------------------------------------------------
# MARKDOWN PROCESSING

class Markdown:
    """
    Tree representation of a markdown file.
    Given a file made of text (prefaced by an optional top heading) folowed by optional sub headings,
    All of the text before the sub headings is stored in header
    Then the sub headings are stored as a list of Markdown classes (each could have their own subheading of lower levels)
    """
    def __init__(self, header: str, headings: list, token_counter):
        self.header = header 
        self.headings = headings
        self.nb_tokens = token_counter(header) + sum(heading.nb_tokens for heading in headings)
    
    def to_string(self):
        result = self.header
        for heading in self.headings:
            result += '\n' + heading.to_string()
        return result

    @staticmethod
    def load(markdown_text, token_counter):
        renderer = MarkdownRenderer(token_counter)
        markdown = mistune.create_markdown(renderer=renderer)
        markdown(markdown_text)
        sections = renderer.finish()

        root = Markdown('', sections, token_counter)
        return root
    
    def to_chunks(self, token_counter, max_tokens):
        if self.nb_tokens < max_tokens:
            # small enough to fit
            return [self.to_string()]
        else:
            # split along headings
            result = text_splitter(self.header, token_counter, max_tokens)
            for heading in self.headings:
                result.extend(heading.to_chunks(token_counter, max_tokens))
            return result

def markdown_splitter(markdown: str, token_counter, max_tokens):
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
    ast = Markdown.load(markdown, token_counter)
    # turn it into a list of chunks of appropriate size
    return ast.to_chunks(token_counter, max_tokens)