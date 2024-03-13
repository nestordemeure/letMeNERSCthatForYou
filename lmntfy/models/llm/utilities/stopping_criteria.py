import torch
from transformers import StoppingCriteria, AutoTokenizer
from typing import List

class StopWordStoppingCriteria(StoppingCriteria):
    """
    Stops the generation if any stopword is encountered.

    NOTE: we have to decode tokens into strings as the same stop word can map to various tokens depending on context

    Inspired by https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/9
    And: https://github.com/outlines-dev/outlines/blob/main/outlines/generate/api.py
    """
    def __init__(self, tokenizer:AutoTokenizer, prompts:List[str], stop_words:List[str]=[], check_every:int=20):
        """
        tokenizer is the tokenizer used by the model
        prompts are the input prompts (to identify new tokens)
        stop_words are the words on which we should stop generation
        check_every is the frequency (in tokens) at which point we should check for stop words
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.input_sizes = [self.tokenizer.encode(prompt, return_tensors="pt").size(-1) for prompt in prompts]
        self.stop_words = stop_words
        self.max_stop_word_size = max(self.tokenizer.encode(word, return_tensors="pt").size(-1) for word in stop_words)
        self.check_every = check_every

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        stop generation if all batch elements contain a stop word in their new tokens AND this is a checking interation
        """
         # gets the dimensions of the problem
        batch_size, seq_len = input_ids.shape
        # keep generating if we have no stopwords
        # or if this is not a checking iteration
        if (len(self.stop_words) == 0) or ((seq_len % self.check_every) != 0): 
            return False
        # iterate on batch elements
        for i in range(batch_size):
            # extract the generated text
            prompt_size = self.input_sizes[i]
            answer_tokens = input_ids[i, prompt_size:]
            # keep only the new tokens that might include a stopword
            max_tokens = 2*self.max_stop_word_size + self.check_every # somewhat arbitrary/pessimistic formula to stay on the safe side
            latest_tokens = answer_tokens[-max_tokens:]
            # decode the generated text
            latest_text = self.tokenizer.decode(latest_tokens, skip_special_tokens=True)
            # if a batch item does NOT contains any stopword then we should keep generating
            if not any((word in latest_text) for word in self.stop_words):
                return False
        return True

    def extract_answers(self, input_ids: torch.LongTensor, strip_stopword=True) -> List[str]:
        """
        Given a batch produced by `generate`,
        remove the prompt (at the beginning of the tokens)
        converts to a list of strings
        cut at the first occurence of a stop word (if any, stripping it by default)
        returns a list of strings
        """
        # gets the dimensions of the problem
        batch_size, seq_len = input_ids.shape
        # iterate on batch elements
        result = []
        for i in range(batch_size):
            # extract the generated text
            prompt_size = self.input_sizes[i]
            answer_tokens = input_ids[i, prompt_size:]
            # decode the generated text
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            # detect the index of the first stopword
            lower_stop_index = len(answer_text)
            for word in self.stop_words:
                    stop_index = answer_text.find(word)
                    if (stop_index != -1):
                        # shift the stop index to include the stopword
                        if not strip_stopword:
                            stop_index += len(word)
                        # records the stop index
                        lower_stop_index = min(stop_index, lower_stop_index)
            # cut the text a the first stopword if any
            answer_text = answer_text[:lower_stop_index]
            # add to results
            result.append(answer_text)
        return result
