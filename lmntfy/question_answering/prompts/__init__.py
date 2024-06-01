from string import Template
import os

#--------------------------------------------------------------------------------------------------
# CLASS

class Prompt:
    """
    Represents a prompt stored on file.
    Variables are represened as `${variable_name}` in the file.

    This implementation has two particulariies:
    * prompts are considered code and thus located relative to this folder
    * one can enable livre reload which lets you tweak prompts *while* the model is running, for easy testing
    """
    def __init__(self, filename:str, live_reload:bool=False):
        self.filename:str = filename
        self.live_reload:bool = live_reload
        self.template:Template = None
        self._load_template()

    def _load_template(self):
        """Load the template from the markdown file."""
        # Directory where this script is located
        script_dir = os.path.dirname(__file__)
        # Assumes the prompt is located relative to THIS code folder
        file_path = os.path.join(script_dir, self.filename)
        with open(file_path, 'r') as file:
            prompt = file.read().strip()
            self.template = Template(prompt)

    def to_string(self, **variables) -> str:
        """Return the prompt with substituted variables."""
        # reload the prompt from file if live-reload is enabled
        if self.live_reload:
            self._load_template()
        return self.template.substitute(variables)

#--------------------------------------------------------------------------------------------------
# PROMPTS

MINIMAL_SYSTEM_PROMPT = Prompt('chat_system_prompt.md')
"""
Basic system prompt.

This prompt is ONLY used to generate a user's question by starting the model's answer with:

```
If I understand you clearly, your question is: "
```
"""

ANSWERING_SYSTEM_PROMPT = Prompt('answering_prompt.md')
"""
Prompt to answer a question
NOTE: 
 * we do a single shot prompt (with an example answer) to ensure proper formating of the answer at the price of a few tokens
 * note that the end of the prompt is ready to accomodate some chunks of information
NOTE: we use "concise and informative" instead of "comprehensive and informative" in our previous iteration of the prompt
"""