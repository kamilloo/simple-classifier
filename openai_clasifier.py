from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.llms import OpenAI
from langchain import PromptTemplate,  LLMChain


llm = OpenAI(
    model_name='text-davinci-003'
)


template = """Classify the text into neutral, negative, or positive. Reply with only one word: Positive, Negative, or Neutral.

Examples:
Text: Big variety of snacks (sweet and savoury) and very good espresso Machiatto with reasonable prices, you can't get wrong if you choose the place for a quick meal or coffee.
Sentiment: Positive.

Text: I got food poisoning
Sentiment: Negative.

Text: {text}
Sentiment:"""

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

