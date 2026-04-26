from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()



# create state
class SentimentState(TypedDict):
    user_prompt:str
    sentiment:str


graph = StateGraph(SentimentState)
