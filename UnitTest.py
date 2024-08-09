import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os
from models import get_embedding_function
from chatbot import chatbot_chat1

def load_test_data(filename):
    df = pd.read_excel(filename)
    return df['Question'].tolist(), df['Expected Answer'].tolist()

def compare_responses(chatbot_response, expected_response, model):
    chatbot_vector = model.embed_query(chatbot_response)
    expected_vector = model.embed_query(expected_response)
    sim_score = cosine_similarity([chatbot_vector], [expected_vector])[0][0]
    return sim_score

def get_gpt4o_metrics(question, expected_answer, model):
    response = chatbot_chat1(question, "gpt-4o")
    similarity_score = compare_responses(response, expected_answer, model)
    return response, similarity_score

def get_gpt35_metrics(question, expected_answer, model):
    response = chatbot_chat1(question, "gpt-3.5-turbo")
    similarity_score = compare_responses(response, expected_answer, model)
    return response, similarity_score

def calculate_metrics_from_cosine(true_labels, scores, threshold=0.833):
    predictions = [1 if score >= threshold else 0 for score in scores]
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    return accuracy, precision, recall, f1, predictions

def plot_performance_metrics(metrics):
    # Plot Performance Metrics
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values_gpt4o = [metrics['Accuracy GPT-4o'], 
                    metrics['Precision GPT-4o'], 
                    metrics['Recall GPT-4o'], 
                    metrics['F1 Score GPT-4o']]
    
    values_gpt35 = [metrics['Accuracy GPT-3.5'], 
                    metrics['Precision GPT-3.5'], 
                    metrics['Recall GPT-3.5'], 
                    metrics['F1 Score GPT-3.5']]
    
    x = range(len(labels))  # x-axis positions for the bars

    # Create bar plot for both models
    plt.figure(figsize=(10, 5))
    bar_width = 0.35  # Width of the bars
    plt.bar(x, values_gpt4o, width=bar_width, label='GPT-4o', color='b', align='center')
    plt.bar([p + bar_width for p in x], values_gpt35, width=bar_width, label='GPT-3.5 Turbo', color='orange', align='center')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Performance Metrics Comparison')
    plt.xticks([p + bar_width / 2 for p in x], labels)
    plt.ylim([0, 1])  # Set y-axis limit from 0 to 1
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def plot_confusion_matrix(true_labels, predictions, model_name):
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Incorrect', 'Correct'], yticklabels=['Incorrect', 'Correct'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

def main():
    model = get_embedding_function()
    test_data_file = os.getenv("TEST_DATA")
    questions, expected_answers = load_test_data(test_data_file)

    metrics = []
    scores_gpt4o = []
    scores_gpt35 = []
    true_labels = []
    similarity_threshold = 0.833

    for question, expected_answer in zip(questions, expected_answers):
        print(f"Processing Question: {question}")

        response_gpt4o, score_gpt4o = get_gpt4o_metrics(question, expected_answer, model)
        response_gpt35, score_gpt35 = get_gpt35_metrics(question, expected_answer, model)

        true_label_gpt4o = 1 if score_gpt4o >= similarity_threshold else 0
        true_label_gpt35 = 1 if score_gpt35 >= similarity_threshold else 0

        true_labels.append((true_label_gpt4o, true_label_gpt35))

        metrics.append({
            'Question': question,
            'Expected Answer': expected_answer,
            'Chatbot Response GPT-4o': response_gpt4o,
            'Cosine Similarity GPT-4o': score_gpt4o,
            'True Label GPT-4o': true_label_gpt4o,
            'Chatbot Response GPT-3.5': response_gpt35,
            'Cosine Similarity GPT-3.5': score_gpt35,
            'True Label GPT-3.5': true_label_gpt35,
        })

        scores_gpt4o.append(score_gpt4o)
        scores_gpt35.append(score_gpt35)

    true_labels_gpt4o, true_labels_gpt35 = zip(*true_labels)

    accuracy_gpt4o, precision_gpt4o, recall_gpt4o, f1_gpt4o, predictions_gpt4o = calculate_metrics_from_cosine(true_labels_gpt4o, scores_gpt4o, similarity_threshold)
    accuracy_gpt35, precision_gpt35, recall_gpt35, f1_gpt35, predictions_gpt35 = calculate_metrics_from_cosine(true_labels_gpt35, scores_gpt35, similarity_threshold)

    print("Performance Metrics:")
    print(f"GPT-4o - Accuracy: {accuracy_gpt4o:.2f}, Precision: {precision_gpt4o:.2f}, Recall: {recall_gpt4o:.2f}, F1 Score: {f1_gpt4o:.2f}")
    print(f"GPT-3.5 Turbo - Accuracy: {accuracy_gpt35:.2f}, Precision: {precision_gpt35:.2f}, Recall: {recall_gpt35:.2f}, F1 Score: {f1_gpt35:.2f}")

    # Store metrics for plotting
    perf_metrics = {
        'Accuracy GPT-4o': accuracy_gpt4o,
        'Precision GPT-4o': precision_gpt4o,
        'Recall GPT-4o': recall_gpt4o,
        'F1 Score GPT-4o': f1_gpt4o,
        'Accuracy GPT-3.5': accuracy_gpt35,
        'Precision GPT-3.5': precision_gpt35,
        'Recall GPT-3.5': recall_gpt35,
        'F1 Score GPT-3.5': f1_gpt35
    }

    # Plot performance metrics
    plot_performance_metrics(perf_metrics)

    # Plot confusion matrices
    plot_confusion_matrix(true_labels_gpt4o, predictions_gpt4o, "GPT-4o")
    plot_confusion_matrix(true_labels_gpt35, predictions_gpt35, "GPT-3.5 Turbo")

    metrics_df = pd.DataFrame(metrics)
    output_file = os.getenv("METRIX")
    metrics_df.to_excel(output_file, index=False)
    print(f"Metrics saved to {output_file}")
    

if __name__ == "__main__":
    main()
    