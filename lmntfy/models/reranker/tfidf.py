import re
from typing import List
from pathlib import Path
from . import Reranker
from ...database.document_splitter import Chunk
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_multiple_whitespaces, remove_stopwords, strip_punctuation, lower_to_unicode, stem_text
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity

# punctuation to be stripped
stripped_punctuation = r"""!"#'()*,-./:;<>?[\]^_`{|}"""
# compiled regexp
RE_stripped_punctuation = re.compile(r'([%s])+' % re.escape(stripped_punctuation), re.UNICODE)
# processing function
def custom_strip_punctuation(s):
    """
    Duplicae of gensim's strip_punctuation function that keep more non-punctuation characters (such as @, $, or %)
    This is useful to avoid breking down some shell output into names that might be things like compiler name.
    See the output [here](https://gitlab.com/NERSC/nersc.gitlab.io/-/blob/main/docs/development/build-tools/spack.md?ref_type=heads&plain=1)
    """
    return RE_stripped_punctuation.sub(" ", s)

# Define a list of preprocessing functions to apply
word_preprocesses = [
    strip_tags,  # remove HTML tags
    #strip_punctuation, # remove punctuation
    custom_strip_punctuation, # remove *most* punctuation
    lower_to_unicode, # set to lowercase
    strip_multiple_whitespaces,  # remove repeating whitespaces
    remove_stopwords,  # remove stopwords
    #stem_text # stem words (NOTE: this seem to somewhat degrade search?)
]

class TFIDFReranker(Reranker):
    def __init__(self, models_folder:Path, name:str='tfidf', device:str='cpu'):
        # initialize the class
        super().__init__(models_folder, name, device)
    
    def _similarity(self, query:str, passage:str) -> float:
        """
        Compute the similarity between a query and a passage
        """
        raise RuntimeError("One should not compute individual similarities with the TFIDFReranker, it requires a corpus of examples. Please use the `similarities` method.")

    def similarities(self, query:str, passages:List[str | Chunk]) -> List[float]:
        """
        Produces a list of similarities for given passages.
        """
        # insures our passages are strings
        if isinstance(passages[0], Chunk):
            passages = [chunk.content for chunk in passages]
        # prepare the corpus
        contents = [preprocess_string(passage, word_preprocesses) for passage in passages]
        dictionary = Dictionary(contents) # id -> word
        corpus = [dictionary.doc2bow(doc) for doc in contents] # document -> word occurence
        tfidf_model = TfidfModel(corpus) # TFIDF computer
        corpus_tfidf = tfidf_model[corpus] # corpus -> tfidf
        similarity_index = MatrixSimilarity(corpus_tfidf, num_features=len(dictionary))
        # prepare the query
        query_doc = preprocess_string(query, word_preprocesses)
        query_bow = dictionary.doc2bow(query_doc)
        query_tfidf = tfidf_model[query_bow]
        # compute the similarity score for each passage
        similarity_scores = list(similarity_index[query_tfidf])
        return similarity_scores