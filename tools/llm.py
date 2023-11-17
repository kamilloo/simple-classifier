from langchain.llms import OpenAI
class Llm:
    def __init__(self):
        self._llm = OpenAI(model_name='text-davinci-003')

    def get_llm(self):

