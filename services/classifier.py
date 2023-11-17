class Classifier:
    def __init__(self, model, llm_chain):
        self.llm_chain = llm_chain
        self.model = model


    def classify(self, text):
        raw_llm_answer = self.llm_chain.run(text)
        llm_answer = raw_llm_answer.lower()
        if "neutral" in llm_answer:
            return 0
        elif "positive" in llm_answer:
            return 1
        elif "negative" in llm_answer:
            return -1
        else:
            raise ValueError(f"Invalid response from the LLM. Response: {raw_llm_answer}")
