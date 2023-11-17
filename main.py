from huggingface_hub.hf_api import HfFolder
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv




from langchain import PromptTemplate,  LLMChain

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

app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    sentiment = classify_fn(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)




