import sys
import os

# Adds the parent directory (Tax-Geek-AI-chatbot) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
# Import the RAG Triad metrics
from deepeval.metrics import (
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    ContextualRelevancyMetric
)
from app import get_response as llm_app

# 1. Define metrics
relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

faithfulness_metric = FaithfulnessMetric(threshold=0.7)

context_relevancy_metric = ContextualRelevancyMetric(threshold=0.1)

metrics = [relevancy_metric, faithfulness_metric, context_relevancy_metric]

# 2. Pull from Confident AI
dataset = EvaluationDataset()
dataset.pull(alias="paye-questions-dataset")

# 3. Create test cases
for golden in dataset.goldens:
    actual_output, retrieval_context = llm_app(golden.input) 
    
    test_case = LLMTestCase(
        input=golden.input,
        expected_output=golden.expected_output,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    dataset.add_test_case(test_case)

# 4. Run an evaluation
evaluate(test_cases=dataset.test_cases, metrics=metrics)