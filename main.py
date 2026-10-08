from DataAgent.agents.Data_agent import Data_Agent
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    response = Data_Agent.invoke(
        {
            "messages": [HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
            "route_response": "",
        }
    )

    print(response["messages"][-1].content)