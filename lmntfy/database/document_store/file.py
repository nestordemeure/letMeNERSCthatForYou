from datetime import datetime
from typing import List, Dict, Any

class File:
    """
    Represent a file, its latest update date and associated chunk indices.
    """

    def __init__(self, update_date: datetime, chunk_indices: List[int] = None):
        """
        Initializes a File instance with a creation date and optional chunk indices.

        Args:
            update_date (datetime): The date and time when the file was last modified.
            chunk_indices (List[int], optional): A list of indices for the associated chunks.
        """
        self.update_date = update_date
        self.chunk_indices = chunk_indices or []

    def add_index(self, chunk_index: int) -> None:
        """
        Adds a chunk index to the list of associated chunk indices.

        Args:
            chunk_index (int): The index to be added to the list of chunk indices.
        """
        self.chunk_indices.append(chunk_index)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the File instance into a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary representation of the File instance.
        """
        return {
            'update_date': self.update_date.isoformat(),
            'chunk_indices': self.chunk_indices
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'File':
        """
        Creates a File instance from a dictionary representation.

        Args:
            data (Dict[str, Any]): A dictionary containing the file data.

        Returns:
            File: A File instance created from the provided dictionary data.
        """
        update_date = datetime.fromisoformat(data['update_date'])
        chunk_indices = data['chunk_indices']
        return File(update_date, chunk_indices)
