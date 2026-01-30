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
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "AI Intern Task"}
)


def retrieve_node(state: AgentState):
    """Fetches relevant segments from the vector database."""
    docs = retriever.invoke(state["question"])
    return {"context_chunks": [doc.page_content for doc in docs]}

def generate_node(state: AgentState):
    """Generates a strictly grounded answer."""
    context_text = "\n\n".join(state["context_chunks"])
    prompt = f"""
    Answer the question using ONLY the provided Context. 
    If the answer is missing, say: "Answer: I cannot find the answer in the provided PDF." and "Score: 0.0"
    Do NOT provide external explanations, bullet points, or reasoning.

    Context: {context_text}
    Question: {state['question']}
    
    Format:
    Answer: [Concise response]
    Score: [0.0 to 1.0]
    """
    response = llm.invoke(prompt).content
    
    try:
        ans_part = response.split("Answer:")[1].split("Score:")[0].strip()
        score_part = float(response.split("Score:")[1].strip())
    except:
        ans_part, score_part = response.strip(), 0.0
        
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