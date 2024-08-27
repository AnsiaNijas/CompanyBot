import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import time
from models import get_openai_embedding_function  # Ensure this function is defined to return an embedding model
from chatbot import chatbot_chat_test  # Ensure this function is implemented correctly
import os

model = get_openai_embedding_function()

def load_test_data(filename):
    df = pd.read_excel(filename)
    return df['Question'].tolist(), df['Expected Answer'].tolist()

def compare_responses(chatbot_response, expected_response, model):
    chatbot_vector = model.embed_query(chatbot_response)
    expected_vector = model.embed_query(expected_response)
    sim_score = cosine_similarity([chatbot_vector], [expected_vector])[0][0]
    return sim_score

def get_gpt_metrics(question, expected_answer, model, engine):
    start_time = time.time()
    response = chatbot_chat_test(question, engine)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    similarity_score = compare_responses(response, expected_answer, model)
    return response, similarity_score, elapsed_time

def calculate_accuracy(similarities, threshold):
    predictions = [1 if sim >= threshold else 0 for sim in similarities]
    true_labels = [1] * len(similarities)  # Assuming all expected answers are correct
    return accuracy_score(true_labels, predictions)

def evaluate_models(questions, expected_answers,output_file):
    model = get_openai_embedding_function()
    results = []  # List to store the results
    gpt4o_similarities = []
    gpt35_similarities = []
    t5_similarities = []
    gpt4o_response_times = []
    gpt35_response_times = []
    t5_response_times = []

    for question, expected_answer in zip(questions, expected_answers):
        # Get metrics for GPT-4o
        gpt4o_response, gpt4o_similarity, gpt4o_time = get_gpt_metrics(question, expected_answer, model, "gpt-4o")
        gpt4o_similarities.append(gpt4o_similarity)
        gpt4o_response_times.append(gpt4o_time)

        # Get metrics for GPT-3.5
        gpt35_response, gpt35_similarity, gpt35_time = get_gpt_metrics(question, expected_answer, model, "gpt-3.5-turbo")
        gpt35_similarities.append(gpt35_similarity)
        gpt35_response_times.append(gpt35_time)

         # Get metrics for T5
        t5_response, t5_similarity, t5_time = get_gpt_metrics(question, expected_answer, model, "t5")
        t5_similarities.append(t5_similarity)
        t5_response_times.append(t5_time)

        # Append the results to the list
        results.append({
            'Question': question,
            'Expected Answer': expected_answer,
            'GPT-4o Response': gpt4o_response,
            'GPT-4o Cosine Similarity': gpt4o_similarity,
            'GPT-4o Response Time (ms)': gpt4o_time,
            'GPT-3.5 Response': gpt35_response,
            'GPT-3.5 Cosine Similarity': gpt35_similarity,
            'GPT-3.5 Response Time (ms)': gpt35_time,
            'T-5 Response': t5_response,
            'T-5 Cosine Similarity': t5_similarity,
            'T-5 Response Time (ms)': t5_time,
        })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Save results to an Excel file
    results_df.to_excel(output_file, index=False)

    # Calculate performance metrics
    threshold = 0.85  # Set an appropriate threshold based on your requirements
    accuracy_gpt4o = calculate_accuracy(gpt4o_similarities, threshold)
    accuracy_gpt35 = calculate_accuracy(gpt35_similarities, threshold)
    accuracy_t5 = calculate_accuracy(t5_similarities, threshold)

    # Print results
    print(f"GPT-4o Accuracy: {accuracy_gpt4o:.2f}")
    print(f"GPT-3.5 Accuracy: {accuracy_gpt35:.2f}")
    print(f"T-5 Accuracy: {accuracy_t5:.2f}")

    print(f"Average Semantic Similarity (GPT-4o): {np.mean(gpt4o_similarities):.2f}")
    print(f"Average Semantic Similarity (GPT-3.5): {np.mean(gpt35_similarities):.2f}")
    print(f"Average Semantic Similarity (T-5): {np.mean(t5_similarities):.2f}")

    print(f"Average Response Time (GPT-4o): {np.mean(gpt4o_response_times):.2f} ms")
    print(f"Average Response Time (GPT-3.5): {np.mean(gpt35_response_times):.2f} ms")
    print(f"Average Response Time (T-5): {np.mean(t5_response_times):.2f} ms")
    # Call the plot function
    plot_results(accuracy_gpt4o, accuracy_gpt35, accuracy_t5, gpt4o_response_times, gpt35_response_times, t5_response_times,
                 gpt4o_similarities, gpt35_similarities, t5_similarities)

def plot_results(accuracy_gpt4o, accuracy_gpt35, accuracy_t5, gpt4o_response_times, gpt35_response_times, t5_response_times,
                 gpt4o_similarities, gpt35_similarities, t5_similarities):
    # Plotting accuracy comparison
    plt.figure(figsize=(10, 6))
    models = ['GPT-4o', 'GPT-3.5', 'T5']
    accuracies = [accuracy_gpt4o, accuracy_gpt35, accuracy_t5]
    
    sns.barplot(x=models, y=accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
    plt.axhline(0.7, color='r', linestyle='--', label='Threshold (0.85)')
    plt.legend()
    plt.show()

    # Plotting response time comparison
    plt.figure(figsize=(10, 6))
    response_times = [np.mean(gpt4o_response_times), np.mean(gpt35_response_times), np.mean(t5_response_times)]
    
    sns.barplot(x=models, y=response_times)
    plt.title('Model Response Time Comparison')
    plt.ylabel('Average Response Time (ms)')
    plt.show()

    # Plotting similarity comparison
    plt.figure(figsize=(10, 6))
    similarities = [np.mean(gpt4o_similarities), np.mean(gpt35_similarities), np.mean(t5_similarities)]
    
    sns.barplot(x=models, y=similarities)
    plt.title('Average Semantic Similarity Comparison')
    plt.ylabel('Average Cosine Similarity')
    plt.ylim(0, 1)  # Set y-axis limit to 0-1 for similarity
    plt.show()

# Main execution logic
if __name__ == "__main__":
    filename = os.getenv("TEST_DATA")  # Path to your Excel file
    output_file= os.getenv("METRIX")
    questions, expected_answers = load_test_data(filename)
    evaluate_models(questions, expected_answers,output_file)
