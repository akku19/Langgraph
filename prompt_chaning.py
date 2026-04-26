from langgraph.graph import START,StateGraph,END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict


load_dotenv()

model = ChatOpenAI()



class genrateBlog(TypedDict):
    user_prompt:str
    blog_outline:str
    final_blog:str
    blog_eval :str

graph = StateGraph(genrateBlog)

def create_outline(state:genrateBlog)->genrateBlog:
    userprompt = state['user_prompt']
    prompt = f'create outline with title of - {userprompt}'
    outline = model.invoke(prompt).content
    state['blog_outline'] = outline
    return state

def genrate_blog(state:genrateBlog)->genrateBlog:
    outline = state['blog_outline']

    prompt = f'create detailed blog according to ouline - {outline}'
    blogdetails = model.invoke(prompt).content
    state['final_blog'] = blogdetails
    return state

def evalution_of_blog(state:genrateBlog)->genrateBlog:
    outline = state['blog_outline']
    blog_details = state['final_blog']
    blogtitle = state['user_prompt']

    prompt = f'according to title of blog {blogtitle} and outline of title {outline} and detailed blog {blog_details} provide me rate according to this details'
    final_eval = model.invoke(prompt).content
    state['blog_eval']= final_eval
    return state


# add nodes 
graph.add_node("create_outline",create_outline)
graph.add_node("genrate_blog",genrate_blog)
graph.add_node("evalution_of_blog",evalution_of_blog)

# add edges 

graph.add_edge(START,"create_outline")
graph.add_edge('create_outline','genrate_blog')
graph.add_edge('genrate_blog',"evalution_of_blog")
graph.add_edge('evalution_of_blog',END)

# compile graph 

workflow = graph.compile()

# run or excute workflow 
initial_value = {"user_prompt":"what is genrative ai"}


finall_result  = workflow.invoke(initial_value)

# print(finall_result['blog_outline'])

print(finall_result['blog_eval'])



