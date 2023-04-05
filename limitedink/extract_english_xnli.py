"""For our baseline report.
Extract the English-language examples from XNLI dataset, which
can be used as a stepping stone to getting XNLI working with 
LimitedInk
"""

import jsonlines

# Create english dev data set
count = 0
with jsonlines.open("../data/XNLI-1.0/xnli.dev.jsonl", 'r') as reader:
    with jsonlines.open("../data/XNLI-1.0/xnli.dev.en.jsonl", mode='w') as writer:
        for obj in reader:
            if obj['language'] == 'en':
                count += 1
                writer.write(obj)
print(f"Number of english dev examples: {count}")

# Create english test data set
count = 0
with jsonlines.open("../data/XNLI-1.0/xnli.test.jsonl", 'r') as reader:
    with jsonlines.open("../data/XNLI-1.0/xnli.test.en.jsonl", mode='w') as writer:
        for obj in reader:
            if obj['language'] == 'en':
                count += 1
                writer.write(obj)
print(f"Number of english test examples: {count}")