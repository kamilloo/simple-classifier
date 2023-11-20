from chains.llm import Llm

def test_llm():
    llm = Llm()
    assert llm.get_llm() is not None