from langchain import PromptTemplate,  LLMChain as LLMChainInterface
from llms.LlmFactory import LlmFactory

class LlmChain:
    def __init__(self):
        self._llm = LlmFactory().create()

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

