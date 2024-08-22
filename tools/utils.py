from collections import Counter

def get_label_freq_bias(train_targets):
    class_counter = Counter(train_targets)

    C = len(class_counter.keys())
    class_bias = []
    for i in class_counter.keys():
        # class_bias.append((C * class_counter[i]) / len(train_targets))
        class_bias.append((class_counter[i]) / len(train_targets))

    return class_bias



