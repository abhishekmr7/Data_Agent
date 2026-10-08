from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

load_dotenv()


def LLM_Pick(level: str):
    """
    Select a model based on the requested level.
    Groq for low/medium/high, Claude Haiku for 'claude'.
    """

    level = level.lower()

    if level == "low":
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

    elif level == "medium":
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

    elif level == "high":
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

    elif level == "claude":
        llm = ChatAnthropic(model="claude-haiku-5-5", temperature=0.7)

    else:
        raise ValueError(
            "Invalid level. Choose 'low', 'medium', 'high', or 'claude'."
        )

    return llm


if __name__ == "__main__":
    llm_ob = LLM_Pick("claude")
    response = llm_ob.invoke("What is the capital of India?")
    print(response.content)