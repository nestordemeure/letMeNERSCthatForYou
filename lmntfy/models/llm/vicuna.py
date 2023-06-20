import re
#import json
#import tiktoken
#import openai
from . import LanguageModel
from .vicuna_support.inference import chat, get_model#, SimpleChatIO, generate_stream
from .. import retry
#import argparse
#import sys

#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to answer a question
ANSWERING_PROMPT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
                The assistant generates a comprehensive and informative answer for a given question solely based on the provided web Search Results (URL and Extract). \
                The assistant search results together into a coherent answer. \
                The assitant only cites the most relevant results that answer the question accurately. \
                The assistant ends the answer with \'References:\' followed by a bullet list of the relevant urls."

# prompt to get references for an answer
REFERENCE_PROMPT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
                The assistant produces a bullet list of the urls useful to answer the question, \
                sorted from most relevant to least relevant. \
                The assistant keeps the bullet list short, keeping only the relevant urls (there are rarely more than three relevant urls)."

def keep_references_only(input_str):
    """keep only lines starting with a *, - or number followed by spaces then a url starting in https""" 
    matches=[]
    input_strings=re.split('\*|\n|\d\ ',input_str)
    for string in input_strings:
        if "http" in string: 
            if string not in matches:
                matches.append(string)

   
    #pattern = re.compile(r'^(?:\*|-|\d+)\s+https:.*$', re.MULTILINE)
    #matches = pattern.findall(input_str)
    return '\n'.join(matches)

# prompt to summarize a conversation into its latest question
QUESTION_EXTRACTION_PROMPT_SYSTEM="You are a question extraction system. \
You will be provided the last messages of a conversation between a user of the NERSC supercomputing center and an assistant from its support, ending on a question by the user. \
Your task is to return the user's last question."
QUESTION_EXTRACTION_PROMPT_USER="Return the user's last question, rephrasing it such that it can be understood without the rest of the conversation."

#----------------------------------------------------------------------------------------
# MODEL

class Vicuna(LanguageModel):
    def __init__(self, 
                 args,
                 model_name='vicuna-13b', 
                 context_size=4096):#4096
        super().__init__(model_name, context_size)
        self.args=args
        #self.chatio=SimpleChatIO()
        self.model, self.tokenizer = get_model(self.args.model_path,self.args.device,self.args.num_gpus,self.args.max_gpu_memory,
                                       self.args.load_8bit,self.args.cpu_offloading,self.args.debug)
        #self.tokenizer2 = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.model_tokens_per_message = 4
        self.model_tokens_per_name = -1
        self.upper_answer_size = 500#250
        self.upper_question_size = 200

    def token_counter_messages(self, messages):
        """
        Token counting implementation for list of messages.
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self.model_tokens_per_message
            for key, value in message.items():
                total_tokens += self.token_counter(value)
                if key == "name":
                    total_tokens += self.model_tokens_per_name
        total_tokens += 2
        return total_tokens

    def token_counter(self, text):
        """
        Token counting implementation.
        """
        encoded_text = self.tokenizer.encode(text)
        return len(encoded_text)


    def get_answer_from_query(self,messages):

        system_messages, context_messages, question_message = messages

        question =question_message['content']

        #chunks = [{'role': 'context', 'content': str(chunk)} for chunk in chunks]
       
        extra= system_messages[0]['content']#'Given the following information: '
        for i in range(len(context_messages)):
            extra += context_messages[i]['content']
        extra += system_messages[1]['content']#'. Answer the following question in detail and provide all refences: '
 
        prompt = extra + system_messages[2]['content']+question+system_messages[3]['content']
        

        return self.query(prompt)

    @retry(n=5)
    def query(self, messages, verbose=False):
        """
        Vicuna specific model query and response.
        """        

        response=chat(messages, self.model, self.tokenizer, self.args.device, 
                      self.args.temperature, self.args.repetition_penalty, self.args.max_new_tokens,self.args.debug)
    
        return response

    def get_answer(self, question, chunks, verbose=False):
        
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        # builds the prompt
        system_messages = [{"role": "system", "content": 'Given the following information: '}, 
                           {"role": "system", "content": '. Answer the following question in detail and provide all refences: '},
                           #{"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "},
                           {"role": "system", "content": ANSWERING_PROMPT +" USER: "},
                            {"role": "system", "content": " ASSISTANT:"}]
        context_messages = [{"role": "assistant", "content": str(chunk)} for chunk in chunks]
        question_message = {"role": "user", "content": question}
        messages = system_messages + context_messages + [question_message]
        # keep as many context messages as we can
        while self.token_counter_messages(messages) + self.upper_answer_size > self.context_size:
            if len(messages) > 2:
                # reduce the context size by popping the latest context message
                # (which is likely the least relevant)
                #messages.pop(-2)
                context_messages.pop(-1)
                messages = system_messages + context_messages + [question_message]
                #chunks = chunks[:-1]
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # runs the query
        messages = [system_messages, context_messages, question_message]
        answer = self.get_answer_from_query(messages)
        # adds sources at the end of the query
        if not "References:" in answer:
            answer = self.add_references(question, answer, chunks, verbose=verbose)
        # returns
        return answer
        
    def add_references(self, question, answer, chunks, verbose=False): #TODO adapt for Vicuna
        """
        Adds references to an answer.
        """
        # builds the prompt 

        system_messages = [{"role": "system", "content": 'Given the following information: '}, 
                           {"role": "system", "content": '. Provide url refences: '},
                           {"role": "system", "content": REFERENCE_PROMPT},
                           {"role": "sysytem", "content": " USER: what are the most important three url references used to asnwer the question?"},
                            {"role": "system", "content": " ASSISTANT:"}]
        context_messages = [{"role": "assistant", "content": str(chunk)} for chunk in chunks]        
        
        extra= system_messages[0]['content']
        for i in range(len(context_messages)):
            extra += context_messages[i]['content']
        extra += system_messages[1]['content']
 
        messages = extra + system_messages[2]['content']+".\ The answer is: "+answer+ system_messages[3]['content']+system_messages[4]['content']
        
        #print(messages)
        
        # runs the query
        references = self.query(messages, verbose=verbose)
        #print(references)
        # remove any irrelevant line
        references = keep_references_only(references)
       
        # updates the answer
        answer = f"{answer}\n\nReferences:\n{references}"
        return answer

    def extract_question(self, previous_messages, verbose=False): #TODO adapt for Vicuna
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
        # keep as many context messages as we can
        while self.token_counter_messages(messages) + self.upper_question_size > self.context_size:
            if len(messages) > 3:
                # reduce the context size by popping the oldest context message
                # ensuring there is at least one context message
                #messages.pop(1)
                context_messages.pop(0)
                messages = [system_message] + context_messages + [user_message]
                
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # shortcut if there is only one message
        if len(messages) == 3:
            return previous_messages[-1]['content']
        else:
            return self.query(messages, verbose=verbose)