from deepeval.dataset import EvaluationDataset, Golden

# Define goldens with the ground truth 'expected_output'
goldens = [
    Golden(
        input="What's the PAYE rate for employee earning 100,000 RWF per month?",
        expected_output="For a monthly income of 100,000 RWF, the PAYE tax rate is 0% for the first 60,000 RWF and 10% for the remaining 40,000 RWF, resulting in a tax of 4,000 RWF."
    ),
    Golden(
        input="What does PAYE stand for?",
        expected_output="PAYE stands for Pay As You Earn. It is a statutory deduction from an employee's remuneration."
    )
]

# Create the dataset
dataset = EvaluationDataset(goldens=goldens)

# Push/Overwrite to Confident AI
dataset.push(alias="paye-questions-dataset")