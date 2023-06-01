from abc import ABC, abstractmethod

class Database(ABC):
    def __init__(self, embedder, file_path=None):
        self.embedder = embedder
        if file_path:
            self.load_from_file(file_path)

    @abstractmethod
    def add_document(self, content, source):
        """
        Abstract method for adding a new document to the database.
        """
        pass

    @abstractmethod
    def get_closest_documents(self, input_text, k=1):
        """
        Abstract method for finding the k closest documents to the input_text in the database.
        """
        pass

    @abstractmethod
    def save_to_file(self, file_path):
        pass

    @abstractmethod
    def load_from_file(self, file_path):
        pass
