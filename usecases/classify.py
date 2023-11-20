from services.classifier import Classifier
from chains.llm import LlmChain


class Classify:
    def __init__(self, classifier: Classifier, llm_chain: LlmChain):
        self.llm_chain = llm_chain
        self.classifier = classifier

    def execute(self, review: str):
        raw = self.llm_chain.run(review)
        return self.classifier.classify(raw)
