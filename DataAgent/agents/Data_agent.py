import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage,AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from utils.etl_tools import ETLTools
from utils.llm_pick import LLM_Pick
from models.schema import ETLAgentSchema, RouterSchema, DataAgentSchema
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst

llm = LLM_Pick("medium")
llm_router = llm.with_structured_output(RouterSchema)


#---DataAgent____Graph

def router_node(state: DataAgentSchema):
    question = state.messages[-1].content
    route = llm_router.invoke([
        SystemMessage(content="Decide whether this request needs the ETL agent or the SQL agent."),
        HumanMessage(content=question),
    ])
    return {"route_response": route.answer}


def etl_node(state: DataAgentSchema):
    question = state.messages[-1].content        # your old code sent the literal text "messages"
    result = etl_analyst.invoke({"messages": [HumanMessage(content=question)]})
    return {"messages": [result["messages"][-1]]}   # only the final AIMessage


def sql_node(state: DataAgentSchema):
    question = state.messages[-1].content
    response = sql_analyst.invoke({
        "messages": [],
        "user_question": question,
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": "",
    })
    return {"messages": [AIMessage(content=response["final_answer"])]} 

Data_Agent_graph = StateGraph(DataAgentSchema)

Data_Agent_graph.add_node("router_node", router_node)
Data_Agent_graph.add_node("etl_node", etl_node)
Data_Agent_graph.add_node("sql_node", sql_node)

Data_Agent_graph.add_edge(START, "router_node")


def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "etl":
        return "etl_node"
    elif state.route_response == "sql":
        return "sql_node"
    else:
        raise ValueError(f"Invalid route response: {state.route_response}")


Data_Agent_graph.add_conditional_edges(
    "router_node",
    route_edge,
    {
        "etl_node": "etl_node",
        "sql_node": "sql_node",
    },
)
Data_Agent_graph.add_edge("etl_node", END)
Data_Agent_graph.add_edge("sql_node", END)

Data_Agent=  Data_Agent_graph.compile()

from IPython.display import display, Image
img = Image(Data_Agent.get_graph().draw_mermaid_png())
with open("data_agent_graph.png", "wb") as f:
    f.write(img.data)



if __name__ == "__main__":

    response = Data_Agent.invoke(
        {"messages":[HumanMessage(content="what are the different types of payment methods we have in our database ")],
         "route_response": ""}
    )

    print(response)



