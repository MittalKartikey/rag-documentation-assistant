import asyncio

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


async def evaluate_all_metrics(
    faithfulness_metric,
    relevancy_metric,
    context_precision_metric,
    sample
):

    results = await asyncio.gather(
        faithfulness_metric.single_turn_ascore(sample),
        relevancy_metric.single_turn_ascore(sample),
        context_precision_metric.single_turn_ascore(sample)
    )

    return results


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

    # Embeddings for Answer Relevancy
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        embedding_model
    )

    # Evaluation sample
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=[
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    # Metrics
    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    relevancy_metric = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    context_precision_metric = LLMContextPrecisionWithoutReference(
        llm=evaluator_llm
    )

    # Run all metrics concurrently
    results = asyncio.run(
        evaluate_all_metrics(
            faithfulness_metric,
            relevancy_metric,
            context_precision_metric,
            sample
        )
    )

    return {
        "faithfulness": float(results[0]),
        "answer_relevancy": float(results[1]),
        "context_precision": float(results[2])
    }
