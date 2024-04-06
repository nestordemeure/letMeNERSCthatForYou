from pathlib import Path
from . import Reranker

class NoReranker(Reranker):
    """
    Does nothing.
    """
    def __init__(self, models_folder:Path, name:str='noreranker', device:str='cpu'):
        super().__init__(models_folder, name, device)

    def _similarity(self, query:str, passage:str) -> float:
        """raise on call"""
        raise RuntimeError("The NoReranker class should never actually be called.")