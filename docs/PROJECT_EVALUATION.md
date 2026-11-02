# 🌾 Agri Research AI — Comprehensive System Evaluation & Architecture Documentation

This document evaluates the **Agri Research AI** platform across four core pillars of autonomous AI systems:
1. **Agent Workflow Planning**
2. **Tool Calling & Execution**
3. **State Management**
4. **Self-Correction & Robustness**

---

## 1. System Overview

**Agri Research AI** is an intelligent agricultural research and decision-support platform designed to assist researchers, agronomists, and farmers. It integrates:
- **Retrieval-Augmented Generation (RAG)** over uploaded agronomic research papers using **ChromaDB** and `sentence-transformers/all-MiniLM-L6-v2`.
- **High-throughput LLM reasoning** powered by **Llama 3.3 70B** via the **Groq** LPUs.
- **Autonomous Multi-Tool Agent Orchestration** using **LangGraph** (ReAct framework).
- **Computer Vision Disease Detection** using a pretrained MobileNetV2 image classification model.
- **Interactive Agronomic Analytics & Diagnostic Consoles** in **Streamlit**.

---

## 2. Evaluation Criterion 1: Agent Workflow Planning

Workflow planning determines how an autonomous agent breaks down complex goals, determines intermediate reasoning steps, selects optimal paths, and coordinates specialized sub-modules.

### 2.1 ReAct Architecture & Decomposition
The core agent engine is defined in [`utils/agent_utils.py`](../utils/agent_utils.py) using LangGraph's prebuilt **ReAct** (Reasoning + Acting) loop via `create_react_agent`:

```
User Query ──► [LLM Reasoner: Llama 3.3 70B]
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
     [Final Answer]       [Tool Selection]
                                 │
                     ┌───────────┼───────────┐
                     ▼           ▼           ▼
               ChromaDB RAG  DuckDuckGo   Agronomy
                 Search      Web Search   Knowledge
                     │           │           │
                     └───────────┼───────────┘
                                 ▼
                         [Observation/Result]
                                 │
                                 └──► Back to LLM Reasoning Loop
```

- **Query Decomposition & Chaining**: The agent can process complex, multi-part inquiries by planning sequential actions. For example, for the query:
  > *"Compare my uploaded papers' findings on nitrogen management with current live fertilizer pricing trends online."*
  
  The agent plans a two-stage execution:
  1. Calls `search_uploaded_papers` to retrieve internal experimental data on nitrogen application rates.
  2. Inspects the observation, then calls `search_the_web` to retrieve current market prices for urea / DAP.
  3. Synthesizes both contexts into a comparative agronomic recommendation.

### 2.2 Dual Execution Modes
The platform implements two distinct operational planning patterns in [`pages/9_Agent_Assistant.py`](../pages/9_Agent_Assistant.py):

| Workflow Mode | Orchestration Mechanism | Planning Characteristics | Best Used For |
| :--- | :--- | :--- | :--- |
| **Fully Autonomous** | `run_agent(question)` | End-to-end autonomous ReAct execution; model decides when to stop chaining tools. | Rapid exploratory research, open queries. |
| **Human-in-the-Loop (HITL)** | `start_agent_turn` / `resume_agent_turn` | Interruptive planning with `interrupt_before=["tools"]`; pauses execution right before tool dispatch for human approval. | High-stakes agronomic decisions, API cost control, compliance validation. |

### 2.3 Domain-Specific Pipeline Planning
In addition to the general agent console, domain-specific pages execute tailored pipelines:
- **Vision-to-RAG Pipeline** ([`pages/5_Disease_Detection.py`](../pages/5_Disease_Detection.py)): Image ingestion $\rightarrow$ Computer Vision feature extraction $\rightarrow$ Top-$k$ classification $\rightarrow$ Diagnostic prompt formulation $\rightarrow$ ChromaDB context search $\rightarrow$ Actionable treatment plan.
- **Soil Parametric Pipeline** ([`pages/4_Soil_Analysis.py`](../pages/4_Soil_Analysis.py)): Rule-based nutrient boundary checks ($\text{pH}, \text{N}, \text{P}, \text{K}$) $\rightarrow$ Deficit calculation $\rightarrow$ Guided RAG retrieval for targeted soil amendment.

---

## 3. Evaluation Criterion 2: Tool Calling and Execution

Tool calling evaluation examines how capabilities are defined, parameterized, passed to the model, and safely executed in the environment.

### 3.1 Tool Schema Definitions
Tools are declared using the `@tool` decorator from `langchain_core.tools` in [`utils/agent_utils.py`](../utils/agent_utils.py). Each tool exposes explicit type signatures and detailed docstrings:

```python
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
```

#### Tool Portfolio Matrix:
1. **`search_uploaded_papers`**:
   - **Underlying Engine**: ChromaDB vector store + HuggingFace `all-MiniLM-L6-v2` embeddings.
   - **Input**: Query string.
   - **Output**: Answer synthesized from top-$k$ retrieved chunks accompanied by document source citations.
2. **`search_the_web`**:
   - **Underlying Engine**: `ddgs` (DuckDuckGo Search API).
   - **Input**: Search query string.
   - **Output**: Up to 5 ranked web snippets containing titles, body excerpts, and hyperlinks.
3. **`general_agriculture_knowledge`**:
   - **Underlying Engine**: Direct Groq LLM invocation (`llama-3.3-70b-versatile`).
   - **Input**: Conceptual question.
   - **Output**: General agronomic principles bypassing local vector storage or web queries.
4. **`classify_leaf_image`** ([`utils/vision_utils.py`](../utils/vision_utils.py)):
   - **Underlying Engine**: Pretrained Hugging Face model (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`).
   - **Input**: Image file / tensor.
   - **Output**: Ranked labels with confidence probabilities.

### 3.2 Tool Invocation & Execution Security
- **Native JSON Schema Generation**: LangGraph transforms the Python docstrings and type annotations into standard OpenAI/Groq function-calling schemas.
- **Execution Isolation**: Tool execution occurs inside LangGraph's managed execution cycle. Tool output is wrapped in a `ToolMessage` and returned to the model context.
- **HITL Inspection Gate**: In verified mode, the UI inspects proposed tool calls using `get_pending_tool_calls(thread_id)` and presents the proposed function name and arguments to the user before any execution takes place.

---

## 4. Evaluation Criterion 3: State Management

State management evaluates how context, session memory, intermediate results, and persistent data stores are maintained across requests and user interactions.

```
┌─────────────────────────────────────────────────────────────┐
│                    State Architecture                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Graph State (LangGraph)                                  │
│    MessagesState: [HumanMessage, AIMessage, ToolMessage...] │
│                                                             │
│ 2. Checkpointer State (MemorySaver)                         │
│    Thread-keyed checkpoints: thread_id = UUID               │
│                                                             │
│ 3. UI Session State (Streamlit st.session_state)            │
│    - agent_history       - hitl_thread_id                   │
│    - pending_tools       - chat_history                     │
│    - soil_*, weather_*   - report_* metrics                 │
│                                                             │
│ 4. Vector & File Persistence (Disk)                         │
│    - chroma_db/ (SQLite + Parquet vector index)             │
│    - data/uploaded_pdfs/ (Raw PDF source documents)         │
│                                                             │
│ 5. Singleton Model Caching (Memory)                         │
│    - _llm, _embeddings, _vectorstore, _processor, _model    │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 LangGraph State & Checkpointing
- **`MessagesState`**: LangGraph maintains the conversation history as an append-only list of messages (`list[BaseMessage]`).
- **`MemorySaver` Checkpointer**: In HITL mode, an in-memory checkpointer (`_hitl_checkpointer = MemorySaver()`) persists execution snapshots keyed by a unique `thread_id`.
- **Interrupt & Resume**: When `interrupt_before=["tools"]` triggers, the graph state is frozen. Once the user clicks *Approve*, `agent.stream(None, config={"configurable": {"thread_id": thread_id}})` resumes execution from the exact checkpoint without re-running previous reasoning steps.

### 4.2 Streamlit Client-Side Session State
Streamlit re-runs scripts from top to bottom on every user interaction. The application utilizes `st.session_state` to prevent state loss:
- `st.session_state.agent_history`: Retains session logs of inquiries, model syntheses, and operational modes.
- `st.session_state.pending_tools`: Holds the unexecuted tool payload during human review.
- `st.session_state.hitl_thread_id`: Maintains session continuity across Streamlit execution cycles.

### 4.3 Persistent Vector Storage
- Local vector database persisted at `chroma_db/` using ChromaDB's SQLite engine.
- Document chunks retain metadata (`source`, `page`) allowing live document index tracking via `get_document_count()` and `list_ingested_sources()`.

### 4.4 Singleton Resource Management
Heavy models (Groq client, HuggingFace embeddings, MobileNetV2 image processor) are instantiated as module-level singletons (`_embeddings`, `_vectorstore`, `_llm`, `_processor`, `_model`). This eliminates redundant initialization overhead between page navigations.

---

## 5. Evaluation Criterion 4: Self-Correction and Robustness

Self-correction evaluates the platform's ability to detect failures, prevent hallucinations, recover from missing data, and gracefully handle edge cases.

### 5.1 Anti-Hallucination & Grounding Guardrails
In RAG pipelines, LLMs frequently fabricate answers when information is absent. Agri Research AI implements strict prompt-level guardrails in [`utils/rag_utils.py`](../utils/rag_utils.py):

```text
You are an agricultural research assistant. Answer the user's question
using ONLY the research context provided below. Be precise, cite specific
findings where possible, and clearly say if the context does not contain
enough information to answer.
```

- When ChromaDB contains insufficient information, the model explicitly acknowledges the gap rather than hallucinating agronomic facts.
- Every retrieved context excerpt includes citation metadata (`source` filename and `page` number), allowing human verification.

### 5.2 Dynamic Tool Fallbacks & Guardrails
- **Empty Vector Index Protection**: If a user runs `search_uploaded_papers` before uploading any documents, `get_document_count()` checks the collection size and immediately returns:
  `"No papers are uploaded yet."`
  This prevents ChromaDB index lookup errors and guides the agent to use `search_the_web` or `general_agriculture_knowledge`.
- **Web Search Resilience**: If DuckDuckGo yields 0 results, `search_the_web` returns `"No web results found."`, signaling the agent to rephrase queries or fall back to baseline knowledge.

### 5.3 Human Intervention as Correction Mechanism
The Human-in-the-Loop workflow serves as an active safety filter:
- If the agent proposes an inappropriate tool or invalid arguments (e.g., query drift), the user can decline execution and trigger a *Reset Verified Session*, preventing erroneous API execution or misleading results.

### 5.4 Defensive Exception Handling
All network calls, vector queries, and vision inference routines are wrapped in defensive `try/except` blocks:
- Missing `GROQ_API_KEY` triggers an explicit `EnvironmentError` with actionable instructions.
- Corrupted images or missing PDF text layers trigger localized error banners without crashing the Streamlit server.

---

## 6. Comprehensive Evaluation Matrix

| Dimension | Implementation Details | Strengths | Opportunities for Enhancement |
| :--- | :--- | :--- | :--- |
| **Agent Workflow Planning** | • LangGraph ReAct Agent<br>• Autonomous & HITL modes<br>• Vision $\rightarrow$ RAG chaining | • Clear separation of autonomous vs. verified paths<br>• Flexible multi-tool chaining | • Add hierarchical supervisor agent for multi-step task decomposition<br>• Introduce automated plan re-evaluation |
| **Tool Calling & Execution** | • `@tool` decorated functions<br>• DuckDuckGo search<br>• ChromaDB RAG<br>• MobileNetV2 CV | • Clean schema definitions<br>• Domain-specific tool portfolio | • Add dynamic weather API tool (OpenWeather)<br>• Add agricultural market price API tool |
| **State Management** | • LangGraph `MemorySaver`<br>• `thread_id` checkpoints<br>• Streamlit `session_state`<br>• Persistent ChromaDB | • Resilient HITL pause/resume<br>• Persistent vector store<br>• Singleton resource reuse | • Upgrade from `MemorySaver` to persistent DB checkpointer (e.g. SQLite / Postgres) for multi-session persistence |
| **Self-Correction & Robustness** | • Anti-hallucination prompts<br>• Empty index guardrails<br>• HITL review gate<br>• Defensive try/except blocks | • Strict citation grounding<br>• Graceful degradation on empty data | • Implement automated RAG reflection (Self-RAG or Corrective RAG / CRAG)<br>• Add query re-writing on low-confidence retrieval |

---

## 7. Architecture Summary & Future Roadmap

```
                                  ┌───────────────────────────┐
                                  │      Streamlit UI         │
                                  │ (Dashboard & Multipage)   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │      LangGraph Agent      │
                                  │   (ReAct / HITL Engine)   │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
    ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
    │    ChromaDB Local RAG     │ │   DuckDuckGo Web Search   │ │  Llama 3.3 70B (Groq LPU)   │
    │  - PyPDFLoader            │ │   - Live Internet data    │ │  - General Agronomy         │
    │  - RecursiveSplitter      │ │   - Commodity prices      │ │  - Direct Reasoning        │
    │  - all-MiniLM-L6-v2       │ │   - Policy & weather      │ │  - Synthesis & Formatting   │
    └───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```

The system provides an enterprise-ready foundation for AI-assisted agronomy, combining local document privacy, live web awareness, deep reasoning, and computer vision.
