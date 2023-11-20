from services.classifier import Classifier
from tools.llm import LlmChain

class Classify:
    def __init__(self, classifier: Classifier, llm_chain: LlmChain):
        self.classifier = classifier
        self.classifier.set_llm_chain()


    def execute(self, text):
        return self.classifier.classify(text)
