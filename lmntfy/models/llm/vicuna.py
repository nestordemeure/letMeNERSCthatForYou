from . import LanguageModel, keep_references_only
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.serve.inference import generate_stream

#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to answer a question
ANSWERING_PROMPT="You are a member of the NERSC supercomputing center's support staff. \
Generate a comprehensive and informative answer for a given question solely based on the provided web Search Results (URL and Extract). \
You must only use information from the provided search results. \
Use an unbiased and journalistic tone. \
Combine search results together into a coherent answer. \
Only cite the most relevant results that answer the question accurately. \
Try and be careful not to go off-topics."

# prompt to get references for an answer
REFERENCE_PROMPT="Produce a bullet list of the urls you found useful to answer the question, \
sorted from most relevant to least relevant. \
Try to keep the bullet list short, keeping *only* the relevant urls (there are rarely more than three relevant urls)."

# prompt to summarize a conversation into its latest question
QUESTION_EXTRACTION_PROMPT_SYSTEM="You are a question extraction system. \
You will be provided the last messages of a conversation between a user of the NERSC supercomputing center and an assistant from its support, ending on a question by the user. \
Your task is to return the user's last question."
QUESTION_EXTRACTION_PROMPT_USER="Return the user's last question, rephrasing it such that it can be understood without the rest of the conversation."

#----------------------------------------------------------------------------------------
# MODEL

class Vicuna(LanguageModel):
    def __init__(self, 
                 model_path,
                 model_name='vicuna-13b', 
                 context_size=2048):
        super().__init__(model_name, context_size)
        self.model, self.tokenizer = load_model(model_path=model_path, device='cuda', num_gpus=1)
        self.conversation_template = get_conversation_template(model_path)
        # TODO update those values with the analysis script
        self.upper_answer_size = 500
        self.upper_question_size = 200

    def messages_to_prompt(self, messages) -> str:
        """Takes an OpenAI-style list of messages and converts it into a Vicuna compatible prompt"""
        # builds a conversation object from the messages
        conv = self.conversation_template.copy()
        for i, message in enumerate(messages):
            if message['role'] == 'system':
                if i > 0: raise RuntimeError("The Vicuna model requiers at most a *single* 'system' prompt passed in first position.")
                conv.system = message['content']
            elif message['role'] == 'user':
                conv.append_message(role='USER', message=message['content'])
            elif message['role'] == 'assistant':
                conv.append_message(role='ASSISTANT', message=message['content'])
            else:
                raise RuntimeError(f"Model only accept 'system', 'user' and 'assistant' roles, not '{message['role']}'")
        # ends on the beginning of the answer message
        # NOTE: testing shows that the answer cannot be primed by a starting message
        conv.append_message(role='ASSISTANT', message=None)
        return conv.get_prompt()

    def token_counter(self, text):
        """
        Token counting implementation.
        """
        encoded_text = self.tokenizer.encode(text)
        return len(encoded_text)

    def query(self, prompt, verbose=False):
        """
        Vicuna specific model query and response.
        """
        # parameters for the generation
        params = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_new_tokens": self.upper_answer_size, # TODO compute the number of tokens left to set this value as high as possible?
            "stop": self.conversation_template.stop_str,
            "stop_token_ids": self.conversation_template.stop_token_ids,
            "echo": verbose,
        }
        # produces an answer as a stream
        stream_response = generate_stream(self.model, self.tokenizer, params, device='cuda', context_len=self.context_size)
        # gets to the last element of the stream
        response = None
        for partial_response in stream_response:
            response = partial_response
        # returns the actual answer
        return response['text'].strip()

    def get_answer(self, question, chunks, verbose=False):
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        # builds the prompt
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "assistant", "content": str(chunk)} for chunk in chunks]
        question_message = {"role": "user", "content": question}
        messages = [system_message] + context_messages + [question_message]
        prompt = self.messages_to_prompt(messages)
        # keep as many context messages as we can
        while self.token_counter(prompt) + self.upper_answer_size > self.context_size:
            if len(messages) > 2:
                # reduce the context size by popping the latest context message
                # (which is likely the least relevant)
                messages.pop(-2)
                chunks = chunks[:-1]
                prompt = self.messages_to_prompt(messages)
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # runs the query
        answer = self.query(prompt, verbose=verbose)
        # adds sources at the end of the query
        answer = self.add_references(question, answer, chunks, verbose=verbose)
        # returns
        return answer
        
    def add_references(self, question, answer, chunks, verbose=False):
        """
        Adds references to an answer.
        """
        # builds the prompt
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "assistant", "content": str(chunk)} for chunk in chunks]
        question_message = {"role": "user", "content": question}
        answer_message = {"role": "assistant", "content": answer}
        reference_message = {"role": "user", "content": REFERENCE_PROMPT}
        messages = [system_message] + context_messages + [question_message, answer_message, reference_message]
        # runs the query
        prompt = self.messages_to_prompt(messages)
        references = self.query(prompt, verbose=verbose)
        # remove any irrelevant lines
        references = keep_references_only(references)
        # updates the answer
        answer = f"{answer}\n\nReferences:\n{references}"
        return answer

    def extract_question(self, previous_messages, verbose=False):
        """
        Extracts the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        # builds the prompt
        system_message = {"role": "system", "content": QUESTION_EXTRACTION_PROMPT_SYSTEM}
        context_messages = [{'role':'assistant', 'content': f"{m['role']}: {m['content']}"} for m in previous_messages]
        user_message = {"role": "user", "content": QUESTION_EXTRACTION_PROMPT_USER}
        messages = [system_message] + context_messages + [user_message]
        prompt = self.messages_to_prompt(messages)
        # keep as many context messages as we can
        while self.token_counter(prompt) + self.upper_question_size > self.context_size:
            if len(messages) > 3:
                # reduce the context size by popping the oldest context message
                # ensuring there is at least one context message
                messages.pop(1)
                prompt = self.messages_to_prompt(messages)
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # shortcut if there is only one message
        if len(messages) == 3:
            return previous_messages[-1]['content']
        else:
            return self.query(prompt, verbose=verbose)