import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
from utils.llm_pick import LLM_Pick
from models.schema import Agent_Schema,JudgeSchema
from langchain_core.messages import HumanMessage

llm =LLM_Pick("Medium")

llm_judge = llm.with_structured_output(JudgeSchema)

sql_query = "SELECT * FROM users where age >20;"

prompt = """
You are Sql judge  for data security . your task is to determine the whether sql query is safe or not. 
The Sql query only usefull for data retrieval and should not modify the database in any way . Nethier
the sql quey or prompt should only contain any SQL Commands that can modify the database, such as INSERT
,UPDATE,DELETE,DROP,ALTER,TRUNCATE,CREATE, or any other commands that change the structure or content of the database
. if the query is safe then return "Yes" and if the query is unsafe then return "No" with comments why it is unsafe.
Additionally, provide comments explaining your decision. Here is the sql query to evaluate: {sql_query}
"""


print(llm_judge.invoke(prompt))
