import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
import json

# Load env
load_dotenv()

# LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# === Agent Definitions ===
agent_definitions = {
    "web": {
        "name": "Web Agent",
        "description": "general knowledge or internet questions",
        "prompt": "You are a helpful AI assistant for general questions. Question: {input}"
    },
    "pdf": {
        "name": "PDF Agent", 
        "description": "document reading and PDF-based queries",
        "prompt": "You are a document analysis AI. Answer using internal document knowledge. Question: {input}"
    },
    "math": {
        "name": "Math Agent",
        "description": "calculations and statistics", 
        "prompt": "You are a math expert. Answer with correct calculations. Question: {input}"
    },
    "summary": {
        "name": "Summarizer Agent",
        "description": "summarizing large text or answers",
        "prompt": "You are a summarizer. Simplify and summarize: {input}"
    }
}

# === Create agent chains using modern RunnableSequence ===
agent_chains = {}
for key, val in agent_definitions.items():
    prompt = PromptTemplate.from_template(val["prompt"])
    chain = prompt | llm
    agent_chains[key] = chain

# === Router using modern approach ===
class RouterDecision(BaseModel):
    destination: str = Field(description="The name of the agent to route to")
    reasoning: str = Field(description="Brief reasoning for the choice")

router_parser = JsonOutputParser(pydantic_object=RouterDecision)

router_template = """
You are a router that decides which expert agent should handle a user question.

Based on the user's question, choose the most appropriate agent from the following options:
{destinations}

{format_instructions}

Question: {input}
"""

destinations_text = "\n".join([f"- {k}: {v['description']}" for k, v in agent_definitions.items()])

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    partial_variables={
        "destinations": destinations_text,
        "format_instructions": router_parser.get_format_instructions()
    }
)

router_chain = router_prompt | llm | router_parser

# === Multi-Agent Chain Implementation ===
def route_and_execute(query_dict):
    """Route the query to the appropriate agent and execute"""
    query = query_dict["input"]
    
    try:
        # Get routing decision
        router_result = router_chain.invoke({"input": query})
        destination = router_result["destination"]
        
        # Execute with the chosen agent
        if destination in agent_chains:
            result = agent_chains[destination].invoke({"input": query})
            return {
                "text": result.content,
                "agent": destination,
                "agent_name": agent_definitions[destination]["name"]
            }
        else:
            # Default to web agent
            result = agent_chains["web"].invoke({"input": query})
            return {
                "text": result.content,
                "agent": "web",
                "agent_name": agent_definitions["web"]["name"]
            }
    except Exception as e:
        # Fallback to web agent on any routing error
        result = agent_chains["web"].invoke({"input": query})
        return {
            "text": result.content,
            "agent": "web",
            "agent_name": agent_definitions["web"]["name"],
            "error": str(e)
        }

# Create the main multi-agent chain
multi_agent_chain = RunnableLambda(route_and_execute)

# === UI ===
st.set_page_config(page_title="🧠 Multi-Agent Research Assistant", layout="wide")
st.title("🧠 Multi-Agent Research Assistant")
st.markdown("Ask anything. Your query will be routed to the right expert agent.")

query = st.text_input("💬 Enter your question:")

if query:
    with st.spinner("Routing to best agent and getting answer..."):
        try:
            result = multi_agent_chain.invoke({"input": query})
            
            agent_name = result["agent_name"]
            final_answer = result["text"]
            
            st.success(f"✅ Answer by `{agent_name}`")
            st.write(final_answer)
            
            # Show routing info in expander
            with st.expander("🔍 Routing Details"):
                st.write(f"**Selected Agent:** {result['agent']}")
                if "error" in result:
                    st.warning(f"Routing error (used fallback): {result['error']}")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your OpenAI API key and try again.")