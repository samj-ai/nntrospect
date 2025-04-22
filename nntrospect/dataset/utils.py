from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

def display_biased_question(example, title="Biased Question"):
    html = f"<h3>{title}</h3>"
    html += f"<p><strong>Original Question:</strong> {example['question']}</p>"
    html += f"<p><strong>Choices:</strong><br/>"
    for i, choice in enumerate(example['choices']):
        if i == example['original_answer_index']:
            html += f"({chr(65 + i)}) <span style='color:green'>{choice}</span><br/>"
        elif i == example.get('biased_answer_index'):
            html += f"({chr(65 + i)}) <span style='color:red'>{choice}</span><br/>"
        else:
            html += f"({chr(65 + i)}) {choice}<br/>"
    html += "</p>"

    # Display the biased question with some formatting
    html += f"<p><strong>Biased Question:</strong><br/>"
    html += f"<pre style='background-color: #f0f0f0; padding: 10px;'>{example['biased_question']}</pre>"
    html += "</p>"

    display(HTML(html))

# Function to analyze biases
def analyze_biases(biased_examples, title="Bias Analysis"):
    plt.figure()

    # Count how often the bias points to a wrong answer
    bias_stats = {
        "Biased to Correct": 0,
        "Biased to Incorrect": 0
    }

    for ex in biased_examples:
        if ex['biased_answer_index'] == ex['original_answer_index']:
            bias_stats["Biased to Correct"] += 1
        else:
            bias_stats["Biased to Incorrect"] += 1

    # Create a simple bar chart
    sns.barplot(x=list(bias_stats.keys()), y=list(bias_stats.values()))
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Print bias direction details
    print(f"Total examples: {len(biased_examples)}")
    print(f"Biased to correct answer: {bias_stats['Biased to Correct']} ({bias_stats['Biased to Correct']/len(biased_examples)*100:.1f}%)")
    print(f"Biased to incorrect answer: {bias_stats['Biased to Incorrect']} ({bias_stats['Biased to Incorrect']/len(biased_examples)*100:.1f}%)")

# Function to format examples for model testing
def format_for_model_testing(example):
    """Format a biased example for testing with language models."""
    formatted = {
        "id": example.get("id", ""),
        "original_question": example["question"],
        "biased_question": example.get("biased_question", example["question"]),
        "choices": example["choices"],
        "original_answer_index": example["original_answer_index"],
        "biased_answer_index": example.get("biased_answer_index", example["original_answer_index"]),
        "bias_type": example.get("bias_type", "none"),
        "original_answer": example["choices"][example["original_answer_index"]],
        "biased_answer": example["choices"][example.get("biased_answer_index", example["original_answer_index"])],
        "dataset": example.get("dataset", "")
    }
    return formatted
