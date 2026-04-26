from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display

# Define state properly
class BmiState(TypedDict):
    weight: float
    height: float
    result: float
    label: str

# Node function
def calculate_bmi(state: BmiState) -> BmiState:
    state["result"] = state["weight"] / (state["height"] ** 2)
    return state


def labled_bmi(state: BmiState) -> BmiState:
    bmi = state["result"]
    if bmi < 18.5:
        label = "Underweight"
    elif 18.5 <= bmi < 24.9:
        label = "Normal weight"
    elif 25 <= bmi < 29.9:
        label = "Overweight"
    else:
        label = "Obesity"
    state["label"] = label
    return state
# Create graph
graph = StateGraph(BmiState)

# Add node
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("label_bmi", labled_bmi)

# Define edges
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmi")
graph.add_edge("label_bmi", END)

# Compile graph
workflow = graph.compile()

# Run graph
input_state = {
    "weight": 70,
    "height": 1.75,
    "result": 0
}

output = workflow.invoke(input_state)
Image(workflow.get_graph().draw_mermaid_png())  # Save graph visualization to a file   
display(Image("workflow.png"))
print(f"BMI Result: {output}")