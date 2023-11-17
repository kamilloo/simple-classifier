class Classify:
    def __init__(self, classifier):
        self.classifier = classifier


    def execute(self, text):
        return self.classifier.Classify(text)
