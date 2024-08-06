import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def calculate_metrics_from_cosine(true_labels, scores, threshold=0.8):
    predictions = [1 if score >= threshold else 0 for score in scores]
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    return accuracy, precision, recall, f1

def main():
    model = get_embedding_function()
    test_data_file = os.getenv("TEST_DATA")
    questions, expected_answers = load_test_data(test_data_file)

    metrics = []
    true_labels = [1 if answer.strip().lower() in ('yes', 'true', '1') else 0 for answer in expected_answers]
    scores_gpt4o = []
    scores_gpt35 = []

    for question, expected_answer in zip(questions, expected_answers):
        print(f"Processing Question: {question}")

        response_gpt4o, score_gpt4o = get_gpt4o_metrics(question, expected_answer, model)
        response_gpt35, score_gpt35 = get_gpt35_metrics(question, expected_answer, model)

        metrics.append({
            'Question': question,
            'Expected Answer': expected_answer,
            'Chatbot Response GPT-4o': response_gpt4o,
            'Cosine Similarity GPT-4o': score_gpt4o,
            'Chatbot Response GPT-3.5': response_gpt35,
            'Cosine Similarity GPT-3.5': score_gpt35,
        })

        scores_gpt4o.append(score_gpt4o)
        scores_gpt35.append(score_gpt35)

    accuracy_gpt4o, precision_gpt4o, recall_gpt4o, f1_gpt4o = calculate_metrics_from_cosine(true_labels, scores_gpt4o)
    accuracy_gpt35, precision_gpt35, recall_gpt35, f1_gpt35 = calculate_metrics_from_cosine(true_labels, scores_gpt35)

    print("Performance Metrics:")
    print(f"GPT-4o - Accuracy: {accuracy_gpt4o:.2f}, Precision: {precision_gpt4o:.2f}, Recall: {recall_gpt4o:.2f}, F1 Score: {f1_gpt4o:.2f}")
    print(f"GPT-3.5 Turbo - Accuracy: {accuracy_gpt35:.2f}, Precision: {precision_gpt35:.2f}, Recall: {recall_gpt35:.2f}, F1 Score: {f1_gpt35:.2f}")

    metrics_df = pd.DataFrame(metrics)
    output_file = os.getenv("METRIX")
    metrics_df.to_excel(output_file, index=False)
    print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    main()
