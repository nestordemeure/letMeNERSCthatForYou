from datetime import datetime

class File:
    def __init__(self, creation_date, vector_database_indices=None):
        # when was the file last modified
        self.creation_date = creation_date
        # list of all database indices
        self.vector_database_indices = vector_database_indices or list()

    def add_index(self, vector_database_index):
        self.vector_database_indices.append(vector_database_index)

    def to_dict(self):
        return {
            'creation_date': self.creation_date.isoformat(),
            'vector_database_indices': self.vector_database_indices
        }

    @staticmethod
    def from_dict(data):
        creation_date = datetime.fromisoformat(data['creation_date'])
        vector_database_indices = data['vector_database_indices']
        return File(creation_date, vector_database_indices)