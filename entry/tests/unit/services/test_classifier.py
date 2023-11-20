import unittest
from services.classifier import Classifier
from enums.review_sentiment import ReviewSentiment as rs



class LlmChain:
    def run(self, text) -> str:
        return text

class LlmChainFail:
    def run(self, text) -> str:
        return 'not recognized'


class TestClassifier(unittest.TestCase):

    def provide_test_cases(self):
        return [
            ('neutral', rs.NEUTRAL.value),
            ('positive', rs.POSITIVE.value),
            ('negative', rs.NEGATIVE.value),
        ]

    def test_classifier(self):

        # GIVEN
        test_cases = self.provide_test_cases()
        llm_chain = LlmChain()
        classifier = Classifier(llm_chain=llm_chain)


        # WHEN
        for (text, expected) in test_cases:
            with self.subTest(text=text, expected=expected):
                # THEN
                self.assertEqual(classifier.classify(text), expected)


    def test_classifier_fail(self):

        # GIVEN
        llm_chain = LlmChainFail()
        classifier = Classifier(llm_chain=llm_chain)

        # THEN
        with self.assertRaises(ValueError):
            # WHEN
            classifier.classify('not recognized')


if __name__ == '__main__':
    unittest.main()