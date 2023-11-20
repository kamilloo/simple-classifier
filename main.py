from flask import Flask, request, jsonify
from flask_cors import CORS
from usecases.classify import Classify

app = Flask(__name__)
CORS(app)

# classify_controller = ClassifyController()

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    classify = Classify()
    sentiment = classify.execute(text)
    return jsonify({'sentiment': sentiment})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
