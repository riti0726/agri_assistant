"""
agent_utils.py
--------------
Turns Research Assistant, Web Search, and general knowledge into
"tools" that Llama 3.3 70B can autonomously choose between and chain
together, instead of the user picking a mode manually.
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from utils.rag_utils import ask_question, get_llm, get_document_count
from utils.rag_utils import web_search as _web_search


@tool
def search_uploaded_papers(question: str) -> str:
    """Search the user's uploaded research papers (ChromaDB) for an answer.
    Use this when the question is about specific research findings, studies,
    or data that might be in uploaded PDFs."""
    if get_document_count() == 0:
        return "No papers are uploaded yet."
    result = ask_question(question)
    sources = ", ".join(s["source"] for s in result["sources"])
    return f"{result['answer']}\n\n(Sources: {sources})"


@tool
def search_the_web(query: str) -> str:
    """Search the live internet for current or general information not
    likely to be in uploaded research papers, e.g. news, prices, current
    events, or general how-to questions."""
    results = _web_search(query, max_results=5)
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"{r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})"
        for r in results
    )


@tool
def general_agriculture_knowledge(question: str) -> str:
    """Answer from general agricultural knowledge when the question doesn't
    need specific papers or live web data — e.g. common crop requirements,
    standard practices, textbook facts."""
    llm = get_llm()
    return llm.invoke(question).content


def get_agent():
    """Build a ReAct-style agent that can pick and chain the tools above."""
    llm = get_llm()
    tools = [search_uploaded_papers, search_the_web, general_agriculture_knowledge]
    return create_react_agent(llm, tools)


def run_agent(question: str) -> str:
    """Run the agent on a single question and return the final answer text."""
    agent = get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_message = result["messages"][-1]
    return final_message.content
# ---------------------------------------------------------------------------
# Human-in-the-loop variant (additive — does not replace run_agent/get_agent)
# ---------------------------------------------------------------------------
from langgraph.checkpoint.memory import MemorySaver

_hitl_checkpointer = MemorySaver()


def get_agent_with_approval():
    """Same tools as get_agent(), but pauses before every tool call so a
    human can approve or reject it first."""
    llm = get_llm()
    tools = [search_uploaded_papers, search_the_web, general_agriculture_knowledge]
    return create_react_agent(
        llm,
        tools,
        checkpointer=_hitl_checkpointer,
        interrupt_before=["tools"],
    )


def start_agent_turn(question: str, thread_id: str):
    """Start a human-verified agent run. Stops automatically if it wants
    to call a tool, and returns the pending tool call(s) for approval."""
    agent = get_agent_with_approval()
    config = {"configurable": {"thread_id": thread_id}}
    agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    return get_pending_tool_calls(thread_id)


def get_pending_tool_calls(thread_id: str):
    """Return the list of tool calls the agent wants to make, or [] if none pending."""
    agent = get_agent_with_approval()
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    if not state.values.get("messages"):
        return []
    last_msg = state.values["messages"][-1]
    return getattr(last_msg, "tool_calls", []) or []


def resume_agent_turn(thread_id: str):
    """Approve pending tool calls and let the agent continue. Returns the
    final answer once done, or the NEXT pending tool calls if it wants to
    call another tool (chaining)."""
    agent = get_agent_with_approval()
    config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    for step in agent.stream(None, config=config, stream_mode="values"):
        final_state = step  # walk through until the graph pauses or finishes

    if not final_state or not final_state.get("messages"):
        return {"status": "error", "message": "Agent produced no output after resuming."}

    last_msg = final_state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    if tool_calls:
        return {"status": "pending", "tool_calls": tool_calls}

    return {"status": "done", "answer": last_msg.content}