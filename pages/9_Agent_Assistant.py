"""
9_🤖_Agent_Assistant.py
-------------------------
Agentic assistant: the model decides for itself whether to search
uploaded papers, search the web, or answer from general knowledge —
and can chain multiple tools before giving a final answer.
"""

import uuid
import streamlit as st
from utils.ui_utils import apply_page_style, page_header
from utils.agent_utils import (
    run_agent, 
    start_agent_turn, 
    resume_agent_turn, 
    get_pending_tool_calls
)

st.set_page_config(page_title="Agent Assistant | Agri Research AI", layout="wide")
apply_page_style()
page_header("Agent Assistant", "Autonomous multi-tool agricultural research agent powered by LangGraph & Llama 3.3 70B.")

# ---------------------------------------------------------------------------
# Interactive Tool Inventory & Capabilities
# ---------------------------------------------------------------------------
st.markdown("#### Available Autonomous Tools")
t_col1, t_col2, t_col3 = st.columns(3)
with t_col1:
    with st.container(border=True):
        st.markdown("**Search Uploaded Papers**")
        st.caption("Searches local ChromaDB vector store for grounded experimental data in your PDFs.")
with t_col2:
    with st.container(border=True):
        st.markdown("**Live Web Search**")
        st.caption("Queries live internet via DuckDuckGo for commodity prices, policy updates, and breaking weather.")
with t_col3:
    with st.container(border=True):
        st.markdown("**General Agronomy Knowledge**")
        st.caption("Direct Llama 3.3 70B reasoning for foundational biology, soil science, and physiology.")

st.write("")

# ---------------------------------------------------------------------------
# Interactive Sample Prompts
# ---------------------------------------------------------------------------
st.markdown("#### Interactive Multi-Tool Inquiry Prompts")
st.caption("Click any complex prompt below to load it into the agent console:")

sample_agent_queries = [
    "Compare my uploaded papers' findings on nitrogen management with current live fertilizer pricing trends online.",
    "Search online for recent fall armyworm outbreaks, then cross-reference with pest management methods in my literature.",
    "What are standard baseline soil pH requirements for soybean, and what are current international trade prices for soybeans?",
]

c_prompts = st.columns(len(sample_agent_queries))
for q_idx, q_text in enumerate(sample_agent_queries):
    with c_prompts[q_idx]:
        if st.button(q_text, key=f"agent_sample_{q_idx}", use_container_width=True):
            st.session_state["agent_query_input"] = q_text

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "hitl_thread_id" not in st.session_state:
    st.session_state.hitl_thread_id = str(uuid.uuid4())

st.write("")
st.subheader("Agent Execution Console")

col_mode, col_info = st.columns([1, 2])
with col_mode:
    execution_mode = st.radio(
        "Execution Mode",
        ["Autonomous (Auto-chain tools)", "Human-in-the-Loop (Approve each tool call)"],
        index=0,
    )

question = st.text_area(
    "Research question for the agent", 
    value=st.session_state.get("agent_query_input", ""),
    placeholder="e.g. Compare what my uploaded papers say about nitrogen use with current fertilizer prices online",
    height=90,
)

# ---------------------------------------------------------------------------
# Autonomous Mode
# ---------------------------------------------------------------------------
if execution_mode == "Autonomous (Auto-chain tools)":
    if st.button("Launch Autonomous Agent", type="primary", disabled=not question.strip()):
        with st.spinner("Agent is reasoning, formulating strategy, and invoking tools..."):
            try:
                answer = run_agent(question)
                st.session_state.agent_history.append({"question": question, "answer": answer, "mode": "Autonomous"})
            except Exception as e:
                st.error(f"Agent failed: {e}")

# ---------------------------------------------------------------------------
# Human-in-the-Loop Mode
# ---------------------------------------------------------------------------
else:
    st.info("In this mode, the agent formulates its plan and asks you for explicit approval before querying the web or vector store.")
    col_start, col_reset = st.columns([1, 4])
    with col_start:
        if st.button("Start Verified Run", type="primary", disabled=not question.strip()):
            st.session_state.hitl_thread_id = str(uuid.uuid4())
            with st.spinner("Agent analyzing request..."):
                try:
                    pending = start_agent_turn(question, st.session_state.hitl_thread_id)
                    st.session_state["pending_tools"] = pending
                    st.session_state["hitl_question"] = question
                except Exception as e:
                    st.error(f"Failed to initialize agent turn: {e}")
    with col_reset:
        if st.button("Reset Verified Session"):
            st.session_state["pending_tools"] = []
            st.session_state.hitl_thread_id = str(uuid.uuid4())
            st.rerun()

    if st.session_state.get("pending_tools"):
        st.warning("Tool Approval Required:")
        for t in st.session_state["pending_tools"]:
            tool_name = getattr(t, "get", lambda k: "")("name") or t.get("name", "Tool")
            tool_args = getattr(t, "get", lambda k: {})("args") or t.get("args", {})
            st.markdown(f"Proposed Tool: **`{tool_name}`** with arguments: `{tool_args}`")
        
        if st.button("Approve & Execute Tool Calls", type="primary"):
            with st.spinner("Executing approved tools and continuing agent reasoning..."):
                try:
                    res = resume_agent_turn(st.session_state.hitl_thread_id)
                    if res["status"] == "done":
                        st.session_state["pending_tools"] = []
                        st.session_state.agent_history.append({
                            "question": st.session_state.get("hitl_question", question), 
                            "answer": res["answer"], 
                            "mode": "Human Verified"
                        })
                        st.rerun()
                    elif res["status"] == "pending":
                        st.session_state["pending_tools"] = res["tool_calls"]
                        st.rerun()
                except Exception as e:
                    st.error(f"Execution error: {e}")

# ---------------------------------------------------------------------------
# History Display
# ---------------------------------------------------------------------------
if st.session_state.agent_history:
    st.divider()
    st.subheader("Agent Session Log")
    for entry in reversed(st.session_state.agent_history):
        with st.container(border=True):
            col_q, col_tag = st.columns([4, 1])
            with col_q:
                st.markdown(f"**Inquiry:** {entry['question']}")
            with col_tag:
                st.markdown(f"<span class='badge-pill badge-agent'>{entry.get('mode', 'Agent')}</span>", unsafe_allow_html=True)
            st.markdown("##### Agent Synthesis:")
            st.write(entry["answer"])