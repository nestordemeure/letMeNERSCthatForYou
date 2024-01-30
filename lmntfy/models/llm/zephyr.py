from pathlib import Path
from . import LanguageModel

class Zephyr(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='zephyr-7b-beta',
                 device='cuda'):
        super().__init__(models_folder / model_name, device=device)
        self.upper_answer_size = 450
        self.upper_question_size = 200
