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

def answer_questions(question_answerer:QuestionAnswerer, questions:List[str]) -> List[str]:
    """run on a handful of test question for quick evaluation purposes"""
    answers = []
    for question in questions:
        print(f"\n> {question}\n")
        answer = question_answerer.get_answer(question, verbose=False)
        print(f"\n{answer}\n")
        answers.append(answer)
    return answers

def chat(question_answerer:QuestionAnswerer) -> List[Dict]:
    """chat with the model"""
    messages = []
    while True:
        question = input("\n> ")
        messages.append({'role':'user', 'content': question})
        answer_message = question_answerer.continue_chat(messages, verbose=False)
        messages.append(answer_message)
        print(f"\n{answer_message['content']}")
    return messages