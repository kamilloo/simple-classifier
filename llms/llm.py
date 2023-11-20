from abc import ABC, abstractmethod

class Llm(ABC):
    @abstractmethod
    def get_llm(self) -> any:
        pass
