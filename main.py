from huggingface_hub.hf_api import HfFolder
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv




app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    sentiment = classify_fn(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)




