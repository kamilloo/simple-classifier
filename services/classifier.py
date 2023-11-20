class Classifier:
    def __init__(self):
        self.llm_chain = None

    def set_llm_chain(self, llm_chain):
        self.llm_chain = llm_chain
        raw_llm_answer = self.llm_chain.run(text)

    def classify(self, text):
        llm_answer = raw_llm_answer.lower()
        if "neutral" in llm_answer:
            return 0
        elif "positive" in llm_answer:
            return 1
        elif "negative" in llm_answer:
            return -1
        else:
            raise ValueError(f"Invalid response from the LLM. Response: {raw_llm_answer}")
