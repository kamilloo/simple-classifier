from usecases.classify import Classify
from services.classifier import Classifier
from chains.llm import LlmChain
from enums.review_sentiment import ReviewSentiment as rs
import unittest


class ClassifyTest(unittest.TestCase):

    def test_classify(self):

        #Given
        classify = Classify()

        #When
        classified = classify.execute(review='positive review')

        #Then
        self.assertEqual(classified, rs.POSITIVE.value)
