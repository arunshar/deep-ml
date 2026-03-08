import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    total = len(labels)
    counts = Counter(labels)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    total_labels = [example[target_attr] for example in examples]
    total_entropy = calculate_entropy(total_labels)

    value_groups = {}
    for example in examples:
        value = example[attr]
        if value not in value_groups:
            value_groups[value] = []
        value_groups[value].append(example)

    weighted_entropy = 0.0
    n = len(examples)

    for subset in value_groups.values():
        subset_labels = [example[target_attr] for example in subset]
        we