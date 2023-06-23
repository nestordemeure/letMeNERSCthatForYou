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

def answer_question(question_answerer:QuestionAnswerer, question) -> str:
    """answers a single question"""
    # gets an answer
    answer = question_answerer.get_answer(question, verbose=False)
    # pretty prints the answer
    markdown_answer = Markdown(answer)
    print()
    Console().print(markdown_answer)
    print()
    return answer

def answer_questions(question_answerer:QuestionAnswerer, questions:List[str]) -> List[str]:
    """run on a handful of test question for quick evaluation purposes"""
    answers = []
    console = Console()
    for question in questions:
        # displays question
        print(f"\n> {question}\n")
        # gets an answer and stores it
        answer = question_answerer.get_answer(question, verbose=False)
        answers.append(answer)
        # pretty prints the answer
        markdown_answer = Markdown(f"\n{answer}\n")
        console.print(markdown_answer)
    return answers

def chat(question_answerer:QuestionAnswerer) -> List[Dict]:
    """chat with the model"""
    messages = []
    console = Console()
    while True:
        # gets question
        question = input("\n> ")
        messages.append({'role':'user', 'content': question})
        # gets an answer and stores it
        answer_message = question_answerer.continue_chat(messages, verbose=False)
        messages.append(answer_message)
        # pretty prints the answer
        markdown_answer = Markdown(f"\n\n{answer_message['content']}\n")
        console.print(markdown_answer)
