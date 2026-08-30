import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def evaluate_rag(question, answer, retrieved_docs, api_key):

    # Groq model used as evaluator
    evaluator_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

    evaluator_llm = LangchainLLMWrapper(
        evaluator_model
    )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=[
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    score = faithfulness_metric.single_turn_score(
        sample
    )

    return score
