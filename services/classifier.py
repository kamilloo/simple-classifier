class Classifier:

    def classify(self, raw_llm_answer):
        llm_answer = raw_llm_answer.lower()
        if "neutral" in llm_answer:
            return 0
        elif "positive" in llm_answer:
            return 1
        elif "negative" in llm_answer:
            return -1
        else:
            raise ValueError(f"Invalid response from the LLM. Response: {raw_llm_answer}")
