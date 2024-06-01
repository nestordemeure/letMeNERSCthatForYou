from typing import List, Dict
from ..models import LanguageModel
from ..database import Database, Chunk
from .reference_cleaning import validate_references, format_reference_list
from .prompts import Prompt, MINIMAL_SYSTEM_PROMPT, ANSWERING_SYSTEM_PROMPT

#----------------------------------------------------------------------------------------
# CLASS

class QuestionAnswerer:
    """
    encapsulate the language model, database, as well as the question answering logic
    """
    def __init__(self, llm: LanguageModel, database: Database):
        # components
        self.llm: LanguageModel = llm
        self.database: Database = database
        # prompts
        self.MINIMAL_SYSTEM_PROMPT: Prompt = MINIMAL_SYSTEM_PROMPT
        self.ANSWERING_SYSTEM_PROMPT: Prompt = ANSWERING_SYSTEM_PROMPT

    async def _extract_question(self, previous_messages:List[Dict], verbose=False) -> str:
        """
        Tries to extract the last question.
        """
        # shortcut for single message conversations
        if len(previous_messages) == 1:
            return previous_messages[-1]['content']
        # builds the messages
        system_message = {"role": "system", "content": self.MINIMAL_SYSTEM_PROMPT.to_string()}
        formatted_discussion = [{**message, 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        messages = [system_message] + formatted_discussion
        # builds the base prompt
        prompt = self.llm.apply_chat_template(messages, nb_tokens_max=self.llm.context_size-self.llm.upper_question_size)
        # prime the model to extract the question
        prompt_question_extraction = prompt + 'If I understand you clearly, your question is: "'
        question = await self.llm.generate(prompt_question_extraction, stopwords=['"'], verbose=verbose)
        return question

    async def _add_references(self, original_prompt: str, chunks: List[Chunk], verbose: bool = False) -> str:
        """
        Generates a reference section for a given prompt using specified documentation chunks.

        Args:
            original_prompt (str): The text generated so far, including the initial prompt.
            chunks (List[Chunk]): A list of documentation data chunks to be referenced.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Returns:
            str: The prompt text appended with a formatted reference section containing URLs.
        """
        # prompt the model to start writing a properly formated reference section
        prompt = original_prompt + "\n\nReferences:\n * <"

        # gets at most one url per chunk
        urls = []
        for _ in chunks:
            # Generate a reference, stops if it is done or ready to start a new reference
            generated_reference = await self.llm.generate(prompt, stopwords=['<'], strip_stopword=False, verbose=verbose)
            # Update the prompt for the next generation
            prompt += generated_reference
            # Extracts the url
            url = generated_reference.split('>', 1)[0]
            urls.append(url)
            # Break the loop if the model is not getting ready to start on a new reference
            if not generated_reference.endswith('<'):
                break

        # Validate and filter URLs.
        valid_urls = validate_references(urls, chunks, original_prompt)
        # Builds a reference section with only the validated URLs.
        reference_section = "References:\n" + format_reference_list(valid_urls)
        return reference_section

    async def _answer_messages(self, discussion:List[Dict[str, str]], chunks:List[Chunk], verbose=False) -> str:
        """
        Chat with the model given the previous messages
        and relevant chunks of the documentation to enrich the chat.
        """
        # builds the messages
        nb_messages_minimum = 3 # keep at least that many messages (when possible)
        nb_relevant_messages = len(chunks) + max(0, len(discussion)-nb_messages_minimum)
        system_message = {"role": "system", "content": self.ANSWERING_SYSTEM_PROMPT.to_string()}
        # formats the chunks
        chunks_messages = [{"role": "system", "content": f"\n{chunk.to_markdown()}", "relevancy": (nb_relevant_messages-i)} for (i,chunk) in enumerate(chunks)]
        # formats the discussion
        if len(discussion) <= nb_messages_minimum:
            discussion_messages = discussion
        else:
            # extracts the last 3 messages, they will stay untouched
            discussion_end = discussion[-3:]
            # extract the first messages and add a relevancy to them
            discussion_start = discussion[:-3]
            discussion_start = [{**message, 'relevancy': i} for (i,message) in enumerate(discussion_start)]
            # assemble the discussion
            discussion_messages = discussion_start + discussion_end
        # assemble the messages
        messages = [system_message] + chunks_messages + discussion_messages

        # turns the messages into a prompt
        prompt = self.llm.apply_chat_template(messages, nb_tokens_max=self.llm.context_size-self.llm.upper_answer_size)

        # generates an answer, stopping at the reference section
        reference_section_titles = ["References:", "**References**:", "Reference(s):", "Sources:", "Ressources:", "Source URL:", "Source URLs:"]
        answer_body = await self.llm.generate(prompt, stopwords=reference_section_titles, verbose=verbose)
        # generate a reference section to go with the answer
        reference_section = await self._add_references(prompt + answer_body, chunks, verbose=verbose)
        # assemble the answer
        return answer_body + '\n\n' + reference_section

    #------------------------------------------------------

    async def answer_messages(self, messages:List[Dict], max_context_size=8, verbose=False) -> Dict:
        """
        Answers the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a message with role 'assistant'.

        Adds `extracted_question` and `documentation_used` fields to the answer in order to help with debugging of bad answers.
        """
        # extracts the search keywords
        keywords = await self._extract_question(messages, verbose=verbose)
        # get a context to help us answer the question
        chunks = self.database.get_closest_chunks(keywords, max_context_size)
        # gets an answer from the model
        answer = await self._answer_messages(messages, chunks, verbose=verbose)
        return {'role': 'assistant', 'content': answer,
                # added debug information
                'extracted_question': keywords,
                'documentation_used': [chunk.to_dict() for chunk in chunks]}

    async def answer_question(self, question:str, max_context_size=8, verbose=False) -> str:
        """shortcut to get the string answer to a single question"""
        # build a single message discussion
        messages = [{'role': 'user', 'content': question}]
        # gets an answer from the model
        answer_message = await self.answer_messages(messages, max_context_size, verbose)
        # returns the answer
        return answer_message['content']
