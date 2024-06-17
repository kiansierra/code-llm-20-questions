from llm_20q.model import LLMQA
import pytest
import os 

os.environ['WANDB_DISABLED'] = 'true'
os.environ["WANDB_MODE"] = "offline"

@pytest.fixture(scope='module')
def llmqa():
    return LLMQA(model_id='meta-llama/Meta-Llama-3-8B-Instruct')

def test_ask(llmqa: LLMQA):
    answer = llmqa.ask([], [], [], do_sample=False)
    assert '?' in answer


