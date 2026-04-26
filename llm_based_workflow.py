from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# print(process.env.OPEN_AI_KEY)

model = ChatOpenAI()

# create state

class LLmstate(TypedDict):
    user_prompt:str
    llm_answer:str

graph = StateGraph(LLmstate)

def llm_qa(state:LLmstate)->LLmstate:
    # extract question from state
    question = state['user_prompt']

    prompt = f'answere the following question {question}'
    answer = model.invoke(prompt).content

    state['llm_answer'] = answer
    return state


# add nodes 
graph.add_node("llm_qa",llm_qa)

# add edges 

graph.add_edge(START,"llm_qa")
graph.add_edge("llm_qa",END)

# compile code 

workflow =graph.compile()

# excute code 

initial_state = {"user_prompt":"what is langchain and example of feature","llm_answer":""}

final_state =workflow.invoke(initial_state)
print(final_state['llm_answer'])






