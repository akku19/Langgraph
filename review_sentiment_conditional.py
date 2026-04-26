from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load env
load_dotenv()

# LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------------
# 1. Sentiment Schema
# -------------------------------
class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(
        description="Sentiment of the text"
    )

# Structured model
model_structured = model.with_structured_output(SentimentSchema)

# -------------------------------
# 2. State Definition
# -------------------------------
class ReviewState(TypedDict):
    review: str
    sentiment: Literal['positive', 'negative', 'neutral']
    diagnosis: dict
    response: str

# -------------------------------
# 3. Graph Functions
# -------------------------------

# Step 1: Find sentiment
def find_sentiment(state: ReviewState):
    prompt = f"Analyze sentiment of this review: {state['review']}"
    result = model_structured.invoke(prompt)
    return {"sentiment": result.sentiment}


# Step 2: Diagnosis for negative/neutral
def run_diagnosis(state: ReviewState):
    prompt = f"""
    Analyze the issue in this review and return a JSON dictionary:
    Review: {state['review']}
    Example output:
    {{
        "issue": "...",
        "severity": "...",
        "category": "..."
    }}
    """
    result = model.invoke(prompt).content
    return {"diagnosis": {"raw": result}}


# Step 3: Positive response
def positive_response(state: ReviewState):
    prompt = f"""
    Write a polite thank-you response for this positive review:
    {state['review']}
    """
    result = model.invoke(prompt).content
    return {"response": result}


# Step 4: Negative response
def negative_response(state: ReviewState):
    prompt = f"""
    Write an apology and resolution response for this negative review:
    {state['review']}
    """
    result = model.invoke(prompt).content
    return {"response": result}


# -------------------------------
# 4. Conditional Routing
# -------------------------------
def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"


# -------------------------------
# 5. Build Graph
# -------------------------------
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("positive_response", positive_response)
graph.add_node("negative_response", negative_response)

# Flow
graph.add_edge(START, "find_sentiment")

graph.add_conditional_edges(
    "find_sentiment",
    check_sentiment,
    {
        "positive_response": "positive_response",
        "run_diagnosis": "run_diagnosis",
    }
)

graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)
graph.add_edge("positive_response", END)

# Compile
workflow = graph.compile()

# -------------------------------
# 6. Run Example
# -------------------------------
initial_state = {
    "review": "I am very sad with the product!"
}

result = workflow.invoke(initial_state)

print("\nFinal Output:\n")
print(result)