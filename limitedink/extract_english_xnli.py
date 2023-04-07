"""Extract the English-language examples from XNLI dataset, which
can be used as a stepping stone to getting XNLI working with 
LimitedInk
"""

import jsonlines
import random
random.seed(0)

# Create english train and val data sets
train_ratio = 0.8   # How much of the dev dataset to turn into training data.  The rest will be val data.
train_count = 0
val_count = 0
dev_count = 0
with jsonlines.open("../data/XNLI-1.0/xnli.dev.jsonl", 'r') as reader:
    english_objs = [obj for obj in reader if obj['language'] == 'en']
random.shuffle(english_objs)
with jsonlines.open("../data/XNLI-1.0/train.jsonl", mode='w') as writer:
    for obj in english_objs[:round(len(english_objs)*train_ratio)]:
        train_count += 1
        dev_count += 1
        writer.write(obj)
with jsonlines.open("../data/XNLI-1.0/val.jsonl", mode='w') as writer:
    for obj in english_objs[round(len(english_objs)*train_ratio):]:
        val_count += 1
        dev_count += 1
        writer.write(obj)
print(f"Number of english train examples: {train_count}")
print(f"Number of english val examples: {val_count}")
print(f"Number of english dev examples (train+val): {dev_count}")

# Create english test data set
test_count = 0
with jsonlines.open("../data/XNLI-1.0/xnli.test.jsonl", 'r') as reader:
    with jsonlines.open("../data/XNLI-1.0/test.jsonl", mode='w') as writer:
        for obj in reader:
            if obj['language'] == 'en':
                test_count += 1
                writer.write(obj)
print(f"Number of english test examples: {test_count}")