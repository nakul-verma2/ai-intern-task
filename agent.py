#this files countains the main logic for the RAG application 
#it can be runned for testing purposes as well
#you can change the desired question in the main function at the bottom

import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END

load_dotenv()


class AgentState(TypedDict):
    question: str
    context_chunks: List[str]
    answer: str
    confidence_score: float

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="agentic_ai_collection"
)
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

llm = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "AI Intern Task"}
)


def retrieve_node(state: AgentState):
    """Broadens retrieval by searching for synonyms and related terms."""
    print("--- NODE: RETRIEVING CONTEXT ---")
    question = state["question"]
    

    expansion_prompt = f"Provide 3 alternative search terms for: '{question}'. Output only the terms."
    variations = llm.invoke(expansion_prompt).content.split("\n")
    

    search_queries = [question] + [v.strip() for v in variations if v.strip()]
    
    all_docs = []
    for q in search_queries:
        all_docs.extend(retriever.invoke(q))
    
    
    unique_chunks = list(set([doc.page_content for doc in all_docs]))[:6]
    
    return {"context_chunks": unique_chunks}
def generate_node(state: AgentState):
    """Generates a detailed, strictly grounded answer with precise formatting."""
    print("--- NODE: GENERATING DETAILED GROUNDED ANSWER ---")
    context_text = "\n\n".join(state["context_chunks"])
    
    
    prompt = f"""
    You are a professional technical assistant. Answer the user's question using ONLY the provided Context.
    
    STRICT RULES:
    1. Provide a COMPREHENSIVE and DETAILED answer based on the available information in the context. 
    2. If the context contains a full explanation (e.g., regarding "Data Management Practices"), include all the key technical points.
    3. If the Context does not contain the answer, say exactly: "Answer: I cannot find the answer in the provided PDF." and "Score: 0.0"
    4. Do NOT explain why you cannot find the answer or use external knowledge.
    5. The Score must be a clean numerical value between 0.0 and 1.0. Do NOT include extra symbols like asterisks, tags, or internal reasoning.

    Context:
    {context_text}
    
    Question: 
    {state['question']}
    
    Required Output Format:
    Answer: [Provide the full, detailed response here]
    Score: [Numerical value only, e.g., 1.0]
    """
    
    response = llm.invoke(prompt).content
    
    
    try:
        if "Answer:" in response and "Score:" in response:
            ans_part = response.split("Answer:")[1].split("Score:")[0].strip()
            score_text = response.split("Score:")[1].strip()
            score_val = "".join(c for c in score_text if c.isdigit() or c == '.')
            score_part = float(score_val)
        else:
            ans_part = response.strip()
            score_part = 0.0
    except Exception:
        ans_part = response.strip()
        score_part = 0.5
        
    return {"answer": ans_part, "confidence_score": score_part}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()


if __name__ == "__main__":
    result = rag_app.invoke({"question": "What is task decomposition in Agentic AI?"})
    print(f"\nANSWER: {result['answer']}\nSCORE: {result['confidence_score']}")