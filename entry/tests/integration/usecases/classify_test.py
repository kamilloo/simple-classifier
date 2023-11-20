from usecases.classify import Classify
from services.classifier import Classifier
from chains.llm import LlmChain
from enums.review_sentiment import ReviewSentiment as rs
import unittest


class ClassifyTest(unittest.TestCase):

    def test_classify(self):
        classifier = Classifier()
        llm_chain = LlmChain()

        classify = Classify(classifier=classifier, llm_chain=llm_chain)

        self.assertEqual(classify.execute(review='positive review'), rs.POSITIVE.value)
