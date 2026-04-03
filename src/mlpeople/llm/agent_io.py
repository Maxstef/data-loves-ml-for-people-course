from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _print_message(msg: Any) -> None:
    """
    Helper to print a single message in a human-readable format.

    Args:
        msg: One of HumanMessage, AIMessage, or ToolMessage
    """
    if isinstance(msg, HumanMessage):
        print(f"[USER]\n{msg.content}\n")
    elif isinstance(msg, AIMessage):
        # Tool call request (ReAct step)
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                print(f"[TOOL CALL] {call['name']}({call['args']})\n")
        # Final LLM response
        elif msg.content.strip():
            print(f"[LLM]\n{msg.content}\n")
    elif isinstance(msg, ToolMessage):
        print(f"[TOOL RESULT] ({msg.name})\n{msg.content}\n")


def pretty_print_agent_result(result: Dict[str, Any]) -> None:
    """
    Pretty-print the full agent execution trace from a completed run.

    Args:
        result: The dictionary returned by an agent invoke call,
                containing a "messages" list.
    """
    print("\n=== AGENT TRACE ===\n")
    for msg in result.get("messages", []):
        _print_message(msg)
    print("=== END TRACE ===\n")


def pretty_print_agent_stream(agent: Any, user_question: str) -> None:
    """
    Pretty-print an agent's output as it streams in real-time.

    Args:
        agent: LangGraph or LangChain agent instance supporting `.stream()`.
        user_question: The question string to send to the agent.
    """
    print("\n=== STREAMING AGENT TRACE ===\n")
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_question}]}
    ):
        for node_data in chunk.values():
            messages = node_data.get("messages", [])
            for msg in messages:
                _print_message(msg)
    print("=== END STREAM ===\n")


# Aliases for backwards compatibility / convenience
print_agent_result = pretty_print_agent_result
print_agent_stream = pretty_print_agent_stream
