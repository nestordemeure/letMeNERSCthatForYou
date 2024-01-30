from rich.markdown import Markdown
from rich.console import Console
from typing import List , Dict
from ..question_answering import QuestionAnswerer

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

def answer_question(question_answerer:QuestionAnswerer, question, verbose=False) -> str:
    """answers a single question"""
    # gets an answer
    answer = question_answerer.get_answer(question, verbose=verbose)
    # pretty prints the answer
    markdown_answer = Markdown(answer)
    print()
    Console().print(markdown_answer)
    print()
    return answer

def answer_questions(question_answerer:QuestionAnswerer, questions:List[str], verbose=False) -> List[str]:
    """run on a handful of test question for quick evaluation purposes"""
    answers = []
    console = Console()
    print()
    for question in questions:
        # displays question
        print(f"> {question}")
        # gets an answer and stores it
        answer = question_answerer.get_answer(question, verbose=verbose)
        answers.append(answer)
        # pretty prints the answer
        markdown_answer = Markdown(answer)
        print()
        console.print(markdown_answer)
        print()
    return answers

def chat(question_answerer:QuestionAnswerer, verbose=False) -> List[Dict]:
    """chat with the model"""
    messages = []
    console = Console()
    print()
    while True:
        # gets question
        question = input("> ")
        messages.append({'role':'user', 'content': question})
        # gets an answer and stores it
        answer_message = question_answerer.chat(messages, verbose=verbose)
        messages.append(answer_message)
        # pretty prints the answer
        markdown_answer = Markdown(answer_message['content'])
        print()
        console.print(markdown_answer)
        print()
