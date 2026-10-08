import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from utils.etl_tools import ETLTools
from utils.llm_pick import LLM_Pick
from models.schema import ETLAgentSchema

@tool
def extract_load_tool(url:str,output_folder:str,format:str):
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """

    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format)



@tool
def transform_load_tool(input_file_path:str,output_folder:str,user_question:str,output_format:str):
    """
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """

    etl_tools = ETLTools()
    top_3_rows = etl_tools.transform_Load(input_file_path)


    llm = LLM_Pick("claude")

    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data. 
            You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path}
            as per the user's question. Do not provide any explanation or comments, only
            the code should be provided. The code should be in a format that can be executed 
            in a Python environment with Pandas installed. 
            Don't write anything else than Pandas Code. \n
            
            Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then 
            write the code to transform and save the data at {output_folder}.
            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {top_3_rows}\n

        """

    response = llm.invoke(prompt).content

    pandas_code = response.strip().strip('```').strip().lstrip('python').strip()

    result = etl_tools.execute_function(pandas_code)

    return f"The data is transformed and saved at {output_folder} in {output_format} format. \n\n Pandas Code Executed: \n {pandas_code} \n\n Execution Result: \n {results}"


#tool KIT
tools = [extract_load_tool,transform_load_tool]

llm = LLM_Pick('high')
llm_bind = llm.bind_tools(tools)



def llm_node(state: ETLAgentSchema):
    system = SystemMessage(content=(
        "You are a Python Data Analyst with tools to extract/load and transform/load data. "
        "Perform the ETL operation the user asks for. Once done, tell the user and stop."
    ))
    response = llm_bind.invoke([system] + state.messages)  # keep the AIMessage, no .content
    return {"messages": state.messages + [response]}



    
def tool_node(state: ETLAgentSchema):
    tool_by_name = {t.name: t for t in tools}
    results = []
    for call in state.messages[-1].tool_calls:
        observation = tool_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=call["id"]))
    return {"messages": state.messages + results}


#---------------Agent Graph-------------------
def is_tool_call(state: ETLAgentSchema):
    return "tool_node" if state.messages[-1].tool_calls else END

etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")
etl_analyst_graph.add_conditional_edges(
    "llm_node", is_tool_call, {"tool_node": "tool_node", END: END}
)
etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()

if __name__ == "__main__":
    # Compile the Graph
    

    # Optional
    from IPython.display import display, Image
    img = Image(etl_analyst.get_graph().draw_mermaid_png())
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(img.data)

    response = etl_analyst.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")]}
    )

#     response = etl_analyst.invoke(
#          {"messages":[HumanMessage(content=f"""
#             I want to transform the data stored in the 'c:\\Data_Agent\\data\\extract\\extracted_data.csv' file 
#             and save the transformed data in the 'c:\\Data_Agent\\data\\transform' folder in the csv format.
#             The transformation should filter the data to show bulbasaur pokemon only.
# """)]}
#     )    

    print(response)
