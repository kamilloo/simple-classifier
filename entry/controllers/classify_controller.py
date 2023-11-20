from flask import Flask as app, request, jsonify

from usecases.classify import Classify


class ClassifyController:

    @app.route('/classify', methods=['POST'])
    def classify(self):
        text = request.json['text']
        classify = Classify()
        sentiment = classify.execute(text)
        return jsonify({'sentiment': sentiment})
