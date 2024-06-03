import asyncio
from rich.markdown import Markdown
from rich.console import Console
from typing import List , Dict
from ..question_answering import QuestionAnswerer
from ..models.llm import LanguageModel

def display_logo():
    """Displays a fancy ascii art logo"""
    lmntfy = "\n\
██╗     ███╗   ███╗███╗   ██╗████████╗███████╗██╗   ██╗\n\
██║     ████╗ ████║████╗  ██║╚══██╔══╝██╔════╝╚██╗ ██╔╝\n\
██║     ██╔████╔██║██╔██╗ ██║   ██║   █████╗   ╚████╔╝ \n\
██║     ██║╚██╔╝██║██║╚██╗██║   ██║   ██╔══╝    ╚██╔╝  \n\
███████╗██║ ╚═╝ ██║██║ ╚████║   ██║   ██║        ██║   \n\
╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝        ╚═╝   "
    print(lmntfy)

async def answer_question(question_answerer:QuestionAnswerer, question, verbose=False) -> str:
    """answers a single question"""
    # gets an answer
    answer = await question_answerer.answer_question(question, verbose=verbose)
    # pretty prints the answer
    markdown_answer = Markdown(answer)
    print()
    Console().print(markdown_answer)
    print()
    return answer

async def answer_questions(question_answerer:QuestionAnswerer, questions:List[str], verbose=False) -> List[str]:
    """run on a handful of test question for quick evaluation purposes"""
    # starts tasks concurently
    tasks = [asyncio.create_task(question_answerer.answer_question(question, verbose=verbose)) for question in questions]
    # displays the answers in order
    answers = []
    console = Console()
    print()
    for question, answer_task in zip(questions, tasks):
        # displays question
        print(f"> {question}")
        # gets an answer and stores it
        answer = await answer_task
        answers.append(answer)
        # pretty prints the answer
        markdown_answer = Markdown(answer)
        print()
        console.print(markdown_answer)
        print()
    return answers

async def chat(question_answerer:QuestionAnswerer, verbose=False) -> List[Dict]:
    """chat with the model, augmenting it with retrieved pieces of documentation."""
    messages = []
    console = Console()
    print()
    while True:
        # gets user input
        question = input("> ")
        messages.append({'role':'user', 'content': question})
        # gets an answer and stores it
        answer_message = await question_answerer.answer_messages(messages, verbose=verbose)
        messages.append(answer_message)
        # pretty prints the answer
        markdown_answer = Markdown(answer_message['content'])
        print()
        console.print(markdown_answer)
        print()

async def basic_chat(model:LanguageModel, verbose=False) -> List[Dict]:
    """chat with the model"""
    messages = []
    console = Console()
    print()
    while True:
        # gets user input
        question = input("> ")
        messages.append({'role':'user', 'content': question})
        # gets an answer and stores it
        prompt = model.apply_chat_template(messages)
        answer = await model.generate(prompt, verbose=verbose)
        answer_message = {'role':'assistant', 'content': answer}
        messages.append(answer_message)
        # pretty prints the answer
        markdown_answer = Markdown(answer_message['content'])
        print()
        console.print(markdown_answer)
        print()