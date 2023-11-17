from abc import ABC, abstractmethod
from

class Llm(ABC):
    @abstractmethod
    def get_llm(self) -> any:
        pass
