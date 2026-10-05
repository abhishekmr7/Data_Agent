from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def LLM_Pick(level: str):
    """
    Select a Groq model based on the requested level.
    """

    if level.lower() == "low":
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7
        )

    elif level.lower() == "medium":
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.7
        )

    elif level.lower() == "high":
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.7
        )

    else:
        raise ValueError(
            "Invalid level. Choose 'low', 'medium', or 'high'."
        )

    return llm


if __name__ == "__main__":
    llm_ob = LLM_Pick("medium")
    response = llm_ob.invoke(
    "What is the capital of India?"
    )
    print(response.content)