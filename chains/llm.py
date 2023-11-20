from llms.openia import OpenAI
from llms.hugging_face import HuggingFace
from langchain import PromptTemplate,  LLMChain as LLMChainInterface
import os
from dotenv import load_dotenv
from enums.llm_source import LlmSource

class LlmChain:
    def __init__(self):

        load_dotenv()

        llm_source = os.getenv("LLM_SOURCE")
        if llm_source == LlmSource.EXTERNAL.value:
            self._llm = OpenAI()
        else:
            self._llm = HuggingFace()

    def run(self, text: str):

        template = """Classify the text into neutral, negative, or positive. Reply with only one word: Positive, Negative, or Neutral.

        Examples:
        Text: Big variety of snacks (sweet and savoury) and very good espresso Machiatto with reasonable prices, you can't get wrong if you choose the place for a quick meal or coffee.
        Sentiment: Positive.

        Text: I got food poisoning
        Sentiment: Negative.

        Text: {text}
        Sentiment:"""

        prompt = PromptTemplate(template=template, input_variables=["text"])

        llm_chain = LLMChainInterface(prompt=prompt, llm=self._llm.get_llm())

        return llm_chain.run(text)

