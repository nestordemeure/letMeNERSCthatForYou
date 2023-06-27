from . import LanguageModel, keep_references_only
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.serve.inference import generate_stream

#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to answer a question
ANSWERING_PROMPT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
                The assistant generates a comprehensive and informative answer for a given question solely based on the provided information (URL and Extract). \
                The assistant combines search results together into a coherent answer. \
                The assitant only cites the most relevant results that answer the question accurately. \
                The assistant ends the answer with \'References:\' followed by a bullet list of the relevant urls.\n"


# prompt to get references for an answer
REFERENCE_PROMPT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
                The assistant produces a bullet list of the urls useful to answer the question, \
                sorted from most relevant to least relevant. \
                The assistant keeps the bullet list short, keeping only the relevant urls (there are rarely more than three relevant urls)."

# prompt to summarize a conversation into its latest question
QUESTION_EXTRACTION_PROMPT_SYSTEM="You are a question extraction system. \
You will be provided the last messages of a conversation between a user of the NERSC supercomputing center and an assistant from its support, ending on a question by the user. \
Your task is to return the user's last question."
QUESTION_EXTRACTION_PROMPT_USER="Return the user's last question, rephrasing it such that it can be understood without the rest of the conversation."

#----------------------------------------------------------------------------------------
# MODEL

class Vicuna(LanguageModel):
    def __init__(self, 
                 models_folder,
                 model_name='vicuna-13b', 
                 context_size=2048):
        super().__init__(models_folder, model_name, context_size)
        model_path = str(models_folder / model_name)
        try:
            # tries to load the model
            self.model, self.tokenizer = load_model(model_path=model_path, device='cuda', num_gpus=1)
        except RuntimeError as e:
            # display a user friendly message in case of failure
            if 'CUDA out of memory' in str(e):
                raise RuntimeError("The model could not be loaded due to a lack of available GPU memory. "
                                   "This may be caused by other processes sharing the GPU. "
                                   "Consider trying to run the code on a new node (login or compute) if your current node might "
                                   "be shared with other GPU-consuming users or resources.")
            raise
        self.conversation = get_conversation_template(model_path) 
        self.upper_answer_size = 300
        self.upper_question_size = 200

    def messages_to_prompt(self, messages) -> str:
        """Takes an OpenAI-style list of messages and converts it into a Vicuna compatible prompt"""
        # builds a conversation object from the messages
        conv = self.conversation.copy()
        # prepare the system message
        conv.system=''
        for message in messages:
            if message['role'] == 'system':
                conv.system += message['content']

        for message in messages:
            if message['role'] == 'user':
                conv.append_message(role='USER', message=message['content'])
            elif message['role'] == 'assistant':
                conv.append_message(role='ASSISTANT', message=message['content'])
            elif message['role'] == 'system':
                continue
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

    def query(self, prompt, prompt_size=None, verbose=False):
        """
        Vicuna specific model query and response.
        """
        # compute the maximum answer size possible
        # see implementation of generate_stream for formula details
        if prompt_size is None: prompt_size = self.token_counter(prompt)
        max_new_tokens = self.context_size - prompt_size - 8

        # parameters for the generation
        params = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop": self.conversation.stop_str,
            "stop_token_ids": self.conversation.stop_token_ids,
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
        system_messages  = [{"role": "system", "content": ANSWERING_PROMPT}]
        system_messages += [{"role": "system", "content": "The assistant answer the users's question based only on the following information:"}]
        context_messages = [{"role": "system", "content": str(chunk)} for chunk in chunks]
        question_message = {"role": "user", "content": question}
        messages = system_messages + context_messages + [question_message]
        prompt = self.messages_to_prompt(messages)
        prompt_size = self.token_counter(prompt)
        # keep as many context messages as we can
        while prompt_size + self.upper_answer_size > self.context_size:
            if len(messages) > 2:
                # reduce the context size by popping the latest context message
                # (which is likely the least relevant)
                chunks.pop(-1)
                context_messages.pop(-1)
                messages = system_messages + context_messages + [question_message]
                prompt = self.messages_to_prompt(messages)
                prompt_size = self.token_counter(prompt)
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # runs the query
        answer = self.query(prompt, prompt_size=prompt_size, verbose=verbose)
        # adds sources at the end of the query
        answer = self.add_references(question, answer, chunks, verbose=verbose)
        # returns
        return answer
        
    def add_references(self, question, answer, chunks, verbose=False):
        """
        Adds references to an answer.
        """
       
        # builds the prompt
        system_messages = [{"role": "system", "content": 'Given the following information: '}, 
                           {"role": "system", "content": '. Provide url refences: '}] 
        system_messages += [{"role": "system", "content": "The assistant answer the users's question based only on the following information:"}] 
        context_messages = [{"role": "system", "content": str(chunk)} for chunk in chunks]
        context_messages += [{"role": "system", "content": REFERENCE_PROMPT}]
        question_message = {"role": "system", "content": ". The question is: "+question}
        answer_message = {"role": "system", "content": "The answer is: "+answer}
        reference_message = {"role": "user", "content": "What are the most important url references used to asnwer the question?"}
        messages = system_messages + context_messages + [question_message, answer_message, reference_message]
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
        prompt_size = self.token_counter(prompt)
        # keep as many context messages as we can
        while prompt_size + self.upper_question_size > self.context_size:
            if len(messages) > 3:
                # reduce the context size by popping the oldest context message
                # ensuring there is at least one context message
                messages.pop(1)
                prompt = self.messages_to_prompt(messages)
                prompt_size = self.token_counter(prompt)
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # shortcut if there is only one message
        if len(messages) == 3:
            return previous_messages[-1]['content']
        else:
            question = self.query(prompt, prompt_size=prompt_size, verbose=verbose)
            # remove an eventual prefix
            if question.startswith('user: '): question = question[6:]
            return question
