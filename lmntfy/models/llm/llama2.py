import sys
from . import LanguageModel, keep_references_only
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast #either Tokenizer version works, using fast for now
#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to answer a question
# this prompt is meant to be followed by a list of chunks
ANSWERING_PROMPT="You are a member of the NERSC supercomputing center's support staff. \
Generate a comprehensive and informative answer for a given question solely based on the provided information (URL and Extract). \
You must only use information from the provided search results. \
Use an unbiased and journalistic tone. \
Combine search results together into a coherent answer. \
Only cite the most relevant results that answer the question accurately. \
Try and be careful not to go off-topics."

# prompt to get references for an answer
REFERENCE_PROMPT="Produce a bullet list of the urls you found useful to answer the question, \
sorted from most relevant to least relevant. \
Keep the bullet list short, keeping *only* the relevant urls (there are rarely more than three relevant urls)."

# prompt to summarize a conversation into its latest question
QUESTION_EXTRACTION_PROMPT_SYSTEM="You are a question extraction system. \
You will be provided the last messages of a conversation between a user of the NERSC supercomputing center and an assistant from its support, ending on a question by the user. \
Your task is to summarize the user's last message."
QUESTION_EXTRACTION_PROMPT_USER="Reframe my last question so that I can forward it to support without the rest of this conversation."

#----------------------------------------------------------------------------------------
# MODEL

class Llama2(LanguageModel):
    def __init__(self, 
                 models_folder,
                 model_name='Llama-2-13b-chat-hf', 
                 context_size=2034):
        super().__init__(models_folder, model_name, context_size)
        model_path = str(models_folder / model_name)
        try:
            # tries to load the model
            self.model = LlamaForCausalLM.from_pretrained(model_path,device_map="auto")
            self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        except RuntimeError as e:
            # display a user friendly message in case of failure
            if 'CUDA out of memory' in str(e):
                raise RuntimeError("The model could not be loaded due to a lack of available GPU memory. "
                                   "This may be caused by other processes sharing the GPU. "
                                   "Consider trying to run the code on a new node (login or compute) if your current node might "
                                   "be shared with other GPU-consuming users or resources.")
            raise
        self.upper_answer_size = 450 #check
        self.upper_question_size = 200 #check

    def messages_to_prompt(self, messages) -> str:
        """Takes an OpenAI-style list of messages and converts it into Llama2 compatible prompt"""
        # builds the prompt string from the messages
        prompt=''
        for message in messages:
            if message['role'] == 'system':
                prompt+="system: "+message['content']+' '
            elif message['role'] == 'user':
                prompt+="user: "+message['content']+' '
            elif message['role'] == 'assistant':
                prompt+="assistant: "+message['content']
            else:
                raise RuntimeError(f"Model only accept 'system', 'user' and 'assistant' roles, not '{message['role']}'")
        return prompt

    def token_counter(self, text):
        """
        Token counting implementation.
        """
        encoded_text = self.tokenizer.encode(text, truncation=False, max_length=sys.maxsize)
        return len(encoded_text)

    def query(self, prompt, prompt_size=None, verbose=False):
        """
        Llama 2 specific model query and response.
        """
        # compute the maximum answer size possible
        # see implementation of generate_stream for formula details
        if prompt_size is None: prompt_size = self.token_counter(prompt)
        max_new_tokens = self.context_size - prompt_size - 8
        # produces an answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=True,temperature=0.6,top_p=0.9)
        reply = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # gets to the last element of the stream
        response = reply.replace(prompt,"").replace("system:","").replace("assistant:","").replace("Answer:","").replace("answer:","")
        return response

    def get_answer(self, question, chunks, verbose=False):
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        # builds the prompt
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "system", "content": f"\n\n{str(chunk)}"} for chunk in chunks]
        question_message = {"role": "user", "content": question}
        messages = [system_message] + context_messages + [question_message]
        prompt = self.messages_to_prompt(messages)
        prompt_size = self.token_counter(prompt)
        # keep as many context messages as we can
        while prompt_size + self.upper_answer_size > self.context_size:
            if len(context_messages) > 1:
                # reduce the context size by popping the latest context message
                # (which is likely the least relevant)
                chunks.pop(-1)
                context_messages.pop(-1)
                messages = [system_message] + context_messages + [question_message]
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
        system_message  = {"role": "system", "content": "URLs available:"}
        context_messages = [{"role": "system", "content": f"\n\n{str(chunk)}"} for chunk in chunks]
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

    def old_extract_question(self, previous_messages, verbose=False):
        """
        Extracts the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        # builds the prompt
        system_message = {"role": "system", "content": QUESTION_EXTRACTION_PROMPT_SYSTEM}
        system_message = {"role": "system", "content": "\n\nConversation:"}
        context_messages = [{'role':'system', 'content': f"\n{m['role']}: {m['content']}"} for m in previous_messages]
        user_message = {"role": "user", "content": QUESTION_EXTRACTION_PROMPT_USER}
        messages = [system_message] + context_messages + [user_message]
        prompt = self.messages_to_prompt(messages)
        prompt_size = self.token_counter(prompt)
        # keep as many context messages as we can
        while prompt_size + self.upper_question_size > self.context_size:
            if len(messages) > 4:
                # reduce the context size by popping the oldest context message
                # ensuring there is at least one context message
                messages.pop(2)
                prompt = self.messages_to_prompt(messages)
                prompt_size = self.token_counter(prompt)
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # shortcut if there is only one message
        if len(messages) == 4:
            return previous_messages[-1]['content']
        else:
            question = self.query(prompt, prompt_size=prompt_size, verbose=verbose)
            # remove an eventual prefix
            if question.startswith('user: '): question = question[6:]
            return question

    def extract_question(self, previous_messages, verbose=False):
        """
        Extracts the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        # builds the prompt
        system_message = {"role": "system", "content": QUESTION_EXTRACTION_PROMPT_SYSTEM}
        user_message = {"role": "user", "content": QUESTION_EXTRACTION_PROMPT_USER}
        messages = [system_message] + previous_messages + [user_message]
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
        last_message = previous_messages[-1]['content']
        if len(messages) == 3:
            return last_message
        else:
            question = self.query(prompt, prompt_size=prompt_size, verbose=verbose)
            # remove an eventual prefix
            if question.startswith('user: '): question = question[6:]
            # combine it with the last nessage so that we cover all of our bases
            question = f"{last_message} ({question})"
            return question

       
    #llama2 -> https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py
    #hf version -> https://huggingface.co/docs/transformers/main/en/model_doc/llama2