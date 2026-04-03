from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable


def query_llm(
    llm: Runnable[str, AIMessage],
    question: str,
    verbose: bool = True,
) -> AIMessage:
    """
    Send a prompt to a LangChain chat model and return the AIMessage response.

    Args:
        llm: Any LangChain chat model implementing `.invoke()` that returns `AIMessage`.
        question (str): The user prompt to send to the model.
        verbose (bool, optional): If True, prints the response content. Defaults to True.

    Returns:
        AIMessage: The complete response from the model.
    """
    response = llm.invoke(question)

    if verbose:
        print(response.content)

    return response


def query_agent(agent: Any, query: str, verbose: bool = True) -> AIMessage:
    """
    Send a user query to a LangChain or LangGraph agent and return the final AIMessage.

    Args:
        agent: The agent instance supporting `.invoke()` method.
        query (str): The user prompt to send to the agent.
        verbose (bool, optional): If True, prints the final AIMessage content. Defaults to True.

    Returns:
        AIMessage: The agent's final response message.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # Typically the last message in "messages" is the agent's response
    final_msg = result["messages"][-1]

    if verbose and hasattr(final_msg, "content"):
        print(final_msg.content)

    return final_msg
