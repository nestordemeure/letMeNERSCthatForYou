import re
from pathlib import Path
from typing import List
from .document_loader import Chunk
from . import FaissDatabase
from ..models import LanguageModel, Embedding
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, stem_text
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity

# the class we will instrument
CORE_VECTORDATABASE_CLASS = FaissDatabase

def keep_alphanumeric(text):
    """
    Keep only alphanumeric characters in the text. Replace other characters with spaces.
    Also turns the text to lowercase.
    """
    return re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()

# Define a list of preprocessing functions to apply
word_preprocesses = [
    strip_tags,  # remove HTML tags
    keep_alphanumeric, # keep only alphanumeric characters
    strip_multiple_whitespaces,  # remove repeating whitespaces
    remove_stopwords,  # remove stopwords
    #stem_text,  # apply stemming
]

class RescoringDatabase(CORE_VECTORDATABASE_CLASS):
    """Takes a vector database class and produce a rescored version of it"""
    def __init__(self, llm:LanguageModel, embedder:Embedding,
                       documentation_folder:Path, database_folder:Path, 
                       min_chunks_per_query=8, update_database=True, name=None):
        super().__init__(llm, embedder, documentation_folder, database_folder, min_chunks_per_query, update_database)

    def get_closest_chunks(self, input_text: str, k: int = 3) -> List[Chunk]:
        """
        returns the (at most) k chunks that contains pieces of text closest to the input_text
        to do so, we first get the closest 2k texts
        then we score them and keep the best k ones
        """
        # get original chunks, to be rescored
        chunks = super().get_closest_chunks(input_text, k*2)
        if len(chunks) <= k: return chunks
        # Preprocess the contents
        contents = [preprocess_string(chunk.content, word_preprocesses) for chunk in chunks]
        dictionary = Dictionary(contents) # id -> word
        corpus = [dictionary.doc2bow(doc) for doc in contents] # document -> word occurence
        tfidf_model = TfidfModel(corpus) # TFIDF computer
        corpus_tfidf = tfidf_model[corpus] # corpus -> tfidf
        similarity_index = MatrixSimilarity(corpus_tfidf, num_features=len(dictionary))
        # prepare the query
        query_doc = preprocess_string(input_text, word_preprocesses)
        #print(f"DEBUGGING: query:{query_doc}")
        query_bow = dictionary.doc2bow(query_doc)
        query_tfidf = tfidf_model[query_bow]
        # Pair each chunk with its similarity score and sort
        similarity_scores = list(similarity_index[query_tfidf])
        chunk_scores = [(chunk, score) for chunk, score in zip(chunks, similarity_scores)]
        sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        #for chunk, score in sorted_chunks:
        #    print(f"DEBUGGING: score:{score} url:{chunk.url}")
        # Extract and return the best k chunks
        best_k_chunks = [chunk for chunk, _ in sorted_chunks[:k]]
        return best_k_chunks