import json
import numpy as np
from collections import defaultdict

# Load the dataset
def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

data = load_data("penn-data.json")

# Split dataset into 80% train, 20% test
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# Count transitions and emissions
transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
bi_transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)

for sentence, tags in train_data:
    prev2, prev1 = "<START1>", "<START2>"
    for word, tag in zip(sentence.split(), tags):
        transition_counts[prev2][prev1][tag] += 1
        bi_transition_counts[prev1][tag] += 1
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1
        prev2, prev1 = prev1, tag

# Compute probabilities
def compute_probs(counts):
    probs = {}
    for key1 in counts:
        probs[key1] = {}
        for key2 in counts[key1]:
            total = sum(counts[key1][key2].values())
            probs[key1][key2] = {k: v / total for k, v in counts[key1][key2].items()}
    return probs

transition_probs = compute_probs(transition_counts)
bi_transition_probs = compute_probs(bi_transition_counts)

emission_probs = {}
for tag in emission_counts:
    total = sum(emission_counts[tag].values())
    emission_probs[tag] = {word: count / total for word, count in emission_counts[tag].items()}

# Find most frequent tag for unseen words
most_frequent_tag = max(tag_counts, key=tag_counts.get)

def viterbi(sentence):
    words = sentence.split()
    n = len(words)
    tags = list(tag_counts.keys())
    dp = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    
    # Initialize
    for t1 in tags:
        for t2 in tags:
            dp[1][t1][t2] = (bi_transition_probs.get("<START2>", {}).get(t1, 1e-6) *
                             transition_probs.get("<START1>", {}).get(t1, {}).get(t2, 1e-6) *
                             emission_probs.get(t2, {}).get(words[0], 1e-6))
            backpointer[1][t1][t2] = "<START1>"
    
    # Recursion
    for i in range(1, n):
        for t1 in tags:
            for t2 in tags:
                max_prob, best_t0 = 0, None
                for t0 in tags:
                    prob = (dp[i][t0][t1] *
                            transition_probs.get(t0, {}).get(t1, {}).get(t2, 1e-6) *
                            emission_probs.get(t2, {}).get(words[i], 1e-6))
                    if prob > max_prob:
                        max_prob, best_t0 = prob, t0
                dp[i + 1][t1][t2] = max_prob
                backpointer[i + 1][t1][t2] = best_t0
    
    # Backtracking
    best_t1, best_t2 = max(((t1, t2) for t1 in tags for t2 in tags), key=lambda x: dp[n][x[0]][x[1]])
    best_tags = [best_t2, best_t1]
    for i in range(n - 2, 0, -1):
        best_tags.append(backpointer[i + 2][best_tags[-1]][best_tags[-2]])
    return list(reversed(best_tags))

# Evaluate
correct, total = 0, 0
tag_wise_correct = defaultdict(int)
tag_wise_total = defaultdict(int)

for sentence, actual_tags in test_data:
    predicted_tags = viterbi(sentence)
    for pred, actual in zip(predicted_tags, actual_tags):
        if pred == actual:
            correct += 1
            tag_wise_correct[actual] += 1
        tag_wise_total[actual] += 1
        total += 1

overall_accuracy = correct / total

tag_wise_accuracy = {tag: tag_wise_correct[tag] / tag_wise_total[tag] for tag in tag_wise_total}

print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
for tag, acc in tag_wise_accuracy.items():
    print(f"Tag {tag}: {acc * 100:.2f}%")
'''import json
import random
import numpy as np
from collections import defaultdict, Counter

# Load dataset
with open("penn-data.json", "r") as f:
    data = json.load(f)

# Shuffle and split into train (80%) and test (20%)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]

# Extract words and tags from training data
tag_counts = Counter()
word_counts = Counter()
transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
emission_counts = defaultdict(lambda: defaultdict(int))

# Define special start symbols for the first two words
START1, START2 = "<s1>", "<s2>"

# Training Phase: Compute transition and emission probabilities
for sentence, tags in train_data:
    prev2, prev1 = START1, START2  # Start symbols
    
    for word, tag in zip(sentence.split(), tags):
        tag_counts[tag] += 1
        word_counts[word] += 1
        emission_counts[tag][word] += 1
        transition_counts[prev2][prev1][tag] += 1
        
        prev2, prev1 = prev1, tag  # Shift for next transition

# Normalize transition probabilities
transition_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
for t2 in transition_counts:
    for t1 in transition_counts[t2]:
        total = sum(transition_counts[t2][t1].values())
        for t in transition_counts[t2][t1]:
            transition_probs[t2][t1][t] = transition_counts[t2][t1][t] / total

# Normalize emission probabilities
emission_probs = defaultdict(lambda: defaultdict(float))
for tag in emission_counts:
    total = sum(emission_counts[tag].values())
    for word in emission_counts[tag]:
        emission_probs[tag][word] = emission_counts[tag][word] / total

# Most frequent tag for handling unseen words
most_frequent_tag = tag_counts.most_common(1)[0][0]

# Viterbi Algorithm for Second Order HMM
def viterbi_second_order(sentence):
    words = sentence.split()
    n = len(words)
    states = list(tag_counts.keys())
    
    # Viterbi table: V[i][t1][t2] stores max prob at position i for (prev2, prev1)
    V = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    # Initialization (first word)
    for t1 in states:
        for t2 in states:
            V[0][START1][t1] = transition_probs[START1][START2].get(t1, 1e-6) * \
                               transition_probs[START2][t1].get(t2, 1e-6) * \
                               emission_probs[t1].get(words[0], 1e-6)

    # Recursion
    for i in range(1, n):
        for t1 in states:
            for t2 in states:
                max_prob, best_prev = 0, None
                for t0 in states:
                    prob = V[i-1][t0][t1] * transition_probs[t0][t1].get(t2, 1e-6) * \
                           emission_probs[t2].get(words[i], 1e-6)
                    if prob > max_prob:
                        max_prob, best_prev = prob, t0
                
                V[i][t1][t2] = max_prob
                backpointer[i][t1][t2] = best_prev

    # Backtracking to find the best sequence
    best_tags = []
    max_final_prob, best_t1, best_t2 = 0, None, None
    for t1 in states:
        for t2 in states:
            if V[n-1][t1][t2] > max_final_prob:
                max_final_prob, best_t1, best_t2 = V[n-1][t1][t2], t1, t2

    best_tags.append(best_t2)
    best_tags.append(best_t1)

    for i in range(n-2, 0, -1):
        best_tags.append(backpointer[i+1][best_tags[-1]][best_tags[-2]])

    return list(reversed(best_tags))

# Evaluation
def evaluate():
    total, correct = 0, 0
    tagwise_correct = Counter()
    tagwise_total = Counter()

    for sentence, true_tags in test_data:
        predicted_tags = viterbi_second_order(sentence)
        
        for pred, actual in zip(predicted_tags, true_tags):
            total += 1
            tagwise_total[actual] += 1
            if pred == actual:
                correct += 1
                tagwise_correct[actual] += 1

    overall_accuracy = correct / total
    tagwise_accuracy = {tag: (tagwise_correct[tag] / tagwise_total[tag]) if tagwise_total[tag] > 0 else 0
                        for tag in tagwise_total}

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print("Tag-wise Accuracy:")
    for tag, acc in sorted(tagwise_accuracy.items(), key=lambda x: x[1], reverse=True):
        print(f"{tag}: {acc:.4f}")

# Run Evaluation
evaluate()'''