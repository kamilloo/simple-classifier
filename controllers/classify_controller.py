from flask import Flask as app, request, jsonify

class ClassifyController:
    def __init__(self, model):
        self.model = model

    @app.route('/classify', methods=['POST'])
    def classify(self):
        text = request.json['text']
        sentiment = classify_fn(text)
        return jsonify({'sentiment': sentiment})
