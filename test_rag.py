from app import prompt_rag
from langchain_openai import ChatOpenAI


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question, expected_response):
    response_text = prompt_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text['answer']
    )

    model = ChatOpenAI(model="gpt-4o-mini")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

def test_PAYE():
    assert query_and_validate(
        question="how much should employee with 300000 monthly salary pay",
        expected_response="the employee should pay 54,000 Rwandan francs in tax on their 300,000 FRW monthly salary.",
    )

def test_tax_due_date():
    assert query_and_validate(
        question="what is the due date to pay tax",
        expected_response="The due date to pay tax is within the prescribed time limit provided by law, which is typically 15 days after the tax declaration period arrives.",
    )


