from huggingface_hub.hf_api import HfFolder
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch


HfFolder.save_token("huggingface access key")

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=30,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})

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

def classify(text):
    raw_llm_answer = llm_chain.run(text)
    llm_answer = raw_llm_answer.lower()
    if "neutral" in llm_answer:
        return 0
    elif "positive" in llm_answer:
        return 1
    elif "negative" in llm_answer:
        return -1
    else:
        raise ValueError(f"Invalid response from the LLM. Response: {raw_llm_answer}")



from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    sentiment = classify(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




