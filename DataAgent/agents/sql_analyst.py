import json
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtil
from utils.llm_pick import LLM_Pick

load_dotenv()


def get_conn_details() -> dict:
    return {
        "host": os.environ["host"],
        "port": os.environ.get("port", "5432"),
        "user": os.environ["user"],
        "password": os.environ["password"],
        "dbname": os.environ["database"],
    }


def clean_sql(text: str) -> str:
    """Remove markdown fences and any <think> style noise from the LLM output."""
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE)
    return text.strip()


# -------------------------------------- AI Agent Code --------------------------------------

def curate_ques(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question

    llm = LLM_Pick("low")

    prompt = f"""
    Rewrite the following question so it is clear, specific and unambiguous for
    a database analyst. Return ONLY the rewritten question, with no explanation.

    Question: {user_question}
    """

    response = llm.invoke(prompt).content.strip()

    state.curated_ques = response
    state.messages = state.messages + [HumanMessage(content=response)]

    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curated_question = state.curated_ques

    obj = DatabaseUtil(get_conn_details())
    schema_info = obj.schema_details("public")

    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language
    query into a Postgres SQL query that can be executed on the database. You are provided
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that
    you can understand the structure of the database and generate an accurate SQL query.
    Unless the user explicitly asks for a specific number of rows, always limit the output to 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications and without markdown code fences.

    User's Original Query: {curated_question}

    Database Schema Details:
    {schema_info}
    """

    state.prompt_query_context = prompt

    return state


def generate_sql(state: AgentSchema) -> AgentSchema:
    prompt = state.prompt_query_context

    llm = LLM_Pick("medium")

    generated_sql_query = llm.invoke(prompt).content
    state.generated_sql_query = clean_sql(generated_sql_query)

    return state


def judge_sql(sql_query: str) -> dict:
    """Ask the LLM judge whether the SQL is safe. Always returns {"answer", "comments"}.

    Groq's gpt-oss models sometimes ignore forced tool calls (function calling),
    so we use JSON-schema structured output and fall back to parsing plain JSON.
    Fails closed ("No") if nothing can be parsed.
    """
    llm = LLM_Pick("medium")

    prompt = f"""
    You are an SQL Judge for data security. Your task is to determine whether the SQL query is
    safe or not. The SQL query should only be used for data retrieval and should not modify the
    database in any way. The SQL query should not contain any SQL commands that can modify the
    database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands
    that can change the structure or content of the database. If the SQL query is safe, answer
    "Yes", otherwise answer "No". Also provide comments explaining your decision.

    Respond ONLY with a JSON object in exactly this format, with no other text:
    {{"answer": "Yes" or "No", "comments": "your explanation"}}

    Here's the SQL query to evaluate:
    {sql_query}"""

    # Attempt 1: native JSON-schema structured output
    try:
        judge = llm.with_structured_output(JudgeSchema, method="json_schema")
        return judge.invoke(prompt).model_dump()
    except Exception as e:
        print(f"Structured output failed, falling back to JSON parsing: {e}")

    # Attempt 2: plain call + parse JSON from the text
    try:
        raw = llm.invoke(prompt).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return JudgeSchema(**json.loads(match.group(0))).model_dump()
    except Exception as e:
        print(f"JSON parsing of judge output failed: {e}")

    return {"answer": "No", "comments": "Judge output could not be parsed, so the query was rejected for safety."}


def is_safe_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query

    # Hard, deterministic guard first: must be a single read-only statement.
    stripped = sql_query.strip().rstrip(";").strip()
    forbidden = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy)\b",
        re.IGNORECASE,
    )
    if (
        not re.match(r"^(select|with)\b", stripped, re.IGNORECASE)
        or ";" in stripped
        or forbidden.search(stripped)
    ):
        state.is_safe = "No"
        state.comments = "Rejected by rule-based check: query must be a single read-only SELECT/WITH statement."
        return state

    # LLM judge as a second layer.
    response = judge_sql(sql_query)
    state.is_safe = response["answer"]
    state.comments = response["comments"]

    return state


def canceled_sql(state: AgentSchema) -> AgentSchema:
    comments = state.comments

    state.final_answer = (
        f"The generated SQL query was deemed unsafe to execute. "
        f"The reason provided by the judge is: {comments}. "
        f"Therefore, the SQL query will not be executed."
    )
    state.messages = state.messages + [AIMessage(content=state.final_answer)]

    return state


def execute_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query

    obj = DatabaseUtil(get_conn_details())
    state.sql_query_execution_result = obj.execute_sql(sql_query)

    return state


def represent_final_answer(state: AgentSchema) -> AgentSchema:
    execution_result = state.sql_query_execution_result
    curated_question = state.curated_ques

    llm = LLM_Pick("low")

    prompt = f"""
    You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
    execution result of the SQL query and the user's original question. The final answer should be
    concise, clear, and directly address the user's query. Avoid including any SQL code or technical
    details in the final answer. The final answer should be in a user-friendly format that is easy to
    understand. If the execution result is empty, is an error, or does not provide a clear answer to
    the user's question, explain this in the final answer.

    Here is the execution result: {execution_result}

    Here is the user's original question: {curated_question}
    """

    llm_response = llm.invoke(prompt).content

    state.final_answer = llm_response
    state.messages = state.messages + [AIMessage(content=llm_response)]

    return state


# ------------------------------------------- Graph Building -------------------------------------------

sql_agent_graph = StateGraph(AgentSchema)

# Nodes
sql_agent_graph.add_node("curate_ques", curate_ques)
sql_agent_graph.add_node("prompt_query_context", prompt_query_context)
sql_agent_graph.add_node("generate_sql", generate_sql)
sql_agent_graph.add_node("is_safe_sql", is_safe_sql)
sql_agent_graph.add_node("canceled_sql", canceled_sql)
sql_agent_graph.add_node("execute_sql", execute_sql)
sql_agent_graph.add_node("represent_final_answer", represent_final_answer)

# Edges
sql_agent_graph.add_edge(START, "curate_ques")
sql_agent_graph.add_edge("curate_ques", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")


# Conditional edge function
def is_safe_sql_edge(state: AgentSchema) -> str:
    if state.is_safe.lower() == "yes":
        return "execute_sql"
    return "canceled_sql"


sql_agent_graph.add_conditional_edges(
    "is_safe_sql",
    is_safe_sql_edge,
    {
        "execute_sql": "execute_sql",
        "canceled_sql": "canceled_sql",
    },
)

sql_agent_graph.add_edge("canceled_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)

# Compile the graph
sql_analyst = sql_agent_graph.compile()


if __name__ == "__main__":
    message = " ".join(sys.argv[1:])
    if not message.strip():
        message = input("Enter your question: ")

    # Optional: save the graph diagram (needs internet for mermaid rendering)
    try:
        png = sql_analyst.get_graph().draw_mermaid_png()
        with open("sql_analyst_graph.png", "wb") as f:
            f.write(png)
    except Exception as e:
        print(f"Skipped graph image: {e}")

    input_schema = {
        "messages": [],
        "user_question": message,
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    sql_analyst_response = sql_analyst.invoke(input_schema)

    print(sql_analyst_response["messages"])
    print("********************************")
    print(sql_analyst_response["generated_sql_query"])
    print("********************************")
    print(sql_analyst_response["sql_query_execution_result"])
    print("********************************")
    print(sql_analyst_response["final_answer"])