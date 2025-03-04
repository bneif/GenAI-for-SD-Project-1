import math
import json
import random
import pickle
from collections import Counter, defaultdict

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return ["<s> " + line.strip() + " <e>" for line in file if line.strip()]  # Add start and end tokens

def calculate_perplexity(model, data, n):
    # We implement perplexity via the exponential function to avoid approximating zero by multiplying many small numbers together
    sum = 0
    token_count = 0
    for method in data:
        tokens = method.split()
        for ngram in create_n_grams(tokens, n):
            prefix, word = tuple(ngram[:-1]), ngram[-1]
            probability = model[prefix].get(word, 1e-10)
            sum += math.log(probability)
            token_count += 1
    
    return math.exp(-sum / token_count)


def create_n_grams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def my_model(corpus, n):
    model = defaultdict(Counter)
    for method in corpus:
        tokens = method.split()
        for n_gram in create_n_grams(tokens, n):
            first_n_minus_one_words, last_word = tuple(n_gram[:-1]), n_gram[-1]
            model[first_n_minus_one_words][last_word] += 1

    # Convert word counts to probabilities
    for first_n_minus_one_words, word_counts in model.items():
        total_count = sum(word_counts.values())
        for word in word_counts:
            word_counts[word] /= total_count

    return model


def find_best_n_gram(train_data, eval_data, n_values, train_choice):
    best_n = None
    best_model = None
    best_perplexity = float('inf')
    
    for n in n_values:
        print("Building n="+str(n))
        model = my_model(train_data, n)
        perplexity = calculate_perplexity(model, eval_data, n)
        print(f"n={n}, Perplexity={perplexity}")
        
        # Save each model to a file
        if train_choice == "s":
            model_filename = f"n_gram_model_student_{n}.pkl"
        else:
            model_filename = f"n_gram_model_instructor_{n}.pkl"
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)
        
        if perplexity < best_perplexity:
            best_n = n
            best_model = model
            best_perplexity = perplexity
    
    return best_n, best_model, best_perplexity

def iterative_predictions(model, test_data, n, output_file):

    selected_methods = random.sample(test_data, 100)

    results = {}

    for i, method in enumerate(selected_methods):
        tokens = method.split()
        prediction_sequence = []
        prefix = tuple(tokens[: n - 1])  # Initial prefix of size n-1

        for _ in range(i, len(tokens)):
            # Find the most likely next token
            next_token = max(model[prefix], key=model[prefix].get)
            probability = model[prefix][next_token]

            prediction_sequence.append({"token": next_token, "probability": probability})

            if next_token == "<e>":  # Stop if method closing bracket is found
                break

            # Update prefix
            prefix = tuple(list(prefix[1:]) + [next_token])

        results[f"Method_{i + 1}"] = prediction_sequence

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Prediction results saved to {output_file}")

          
def main():
    random.seed(42)  # Set a random seed for reproducibility
    
    train_choice = input("Will you use the student's training data (s) or the instructor's training data (i)? ").strip().lower()
    if train_choice == "s":
        train_file = "training_student.txt" 
    else:
        train_file = "training_instructor.txt"
    train_data = load_data(train_file)
    
    eval_data = load_data("eval.txt")
    test_data = load_data("test.txt")
    
    corpus = train_data + eval_data + test_data  # Ensuring all tokens are seen during training

    # Build the vocabulariy
    vocabulary = set()
    for method in corpus:
        tokens = method.split()
        # Add each token to our vocabulary
        vocabulary.update(tokens)  
    
    n_values = [3, 5, 7, 9]
    best_n, best_model, best_perplexity = find_best_n_gram(corpus, eval_data, n_values, train_choice)
    
    print("Perplexity for n=" + str(best_n)+" over test set is "+str(calculate_perplexity(best_model, test_data, best_n)))
    
    # Save results based on training choice
    results_file = "results_student_model.json" if train_choice == "s" else "results_instructor_model.json"
    #save_results(results_file, perplexities)
    iterative_predictions(best_model, test_data, best_n, results_file)
    

if __name__ == "__main__":
    main()
