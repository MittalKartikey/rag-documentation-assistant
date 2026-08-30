from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import SingleTurnSample

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def evaluate_rag(question, answer, retrieved_docs, api_key):

    # Evaluator LLM
    evaluator_model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=api_key
    )

    evaluator_llm = LangchainLLMWrapper(
        evaluator_model
    )

    # Same embedding model used by the RAG system
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        embedding_model
    )

    # Build evaluation sample
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=[
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    # 1. Faithfulness
    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    faithfulness_score = faithfulness_metric.single_turn_score(
        sample
    )

    # 2. Answer Relevancy
    relevancy_metric = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    relevancy_score = relevancy_metric.single_turn_score(
        sample
    )

    # 3. Context Precision
    context_precision_metric = LLMContextPrecisionWithoutReference(
        llm=evaluator_llm
    )

    context_precision_score = (
        context_precision_metric.single_turn_score(sample)
    )

    return {
        "faithfulness": float(faithfulness_score),
        "answer_relevancy": float(relevancy_score),
        "context_precision": float(context_precision_score)
    }
