import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationSummaryMemory
import os
from dotenv import load_dotenv

load_dotenv(".env")

st.title("AI Search Engine: Context Aware Agent")

with st.sidebar:
    key = os.getenv("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")

if not key:
    st.warning("Groq API Key required in sidebar")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=key)

arxiv_wrap = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=200)

tool_list = [
    ArxivQueryRun(api_wrapper=arxiv_wrap),
    DuckDuckGoSearchRun(name="WebSearch"),
]

if "mem" not in st.session_state:
    st.session_state.mem = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

agent = initialize_agent(
    tools=tool_list,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state.mem,
    handle_parsing_errors=True,
    verbose=False,
)

if "hist" not in st.session_state:
    st.session_state.hist = []

for m in st.session_state.hist:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if ui := st.chat_input("Type your message"):
    st.session_state.hist.append({"role": "user", "content": ui})
    with st.chat_message("user"):
        st.markdown(ui)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        out = agent.run(ui, callbacks=[cb])
        st.markdown(out)
        st.session_state.hist.append({"role": "assistant", "content": out})
