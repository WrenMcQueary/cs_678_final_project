"""Modify the train.jsonl, val.jsonl, and test.jsonl files in ../data/e-XNLI-just-english/,
removing all but the English elements from them.
"""


import os
import shutil
import random
import glob as gb
from math import floor
import pandas as pd
import json


SOURCE_PATH = "../data/e-XNLI-just-english/"


# Read .jsonl files
with open(os.path.join(SOURCE_PATH, "train.jsonl"), "r", encoding="utf-8") as file:
    train_list = list(file)
    train_list = [json.loads(el) for el in train_list]
with open(os.path.join(SOURCE_PATH, "val.jsonl"), "r", encoding="utf-8") as file:
    val_list = list(file)
    val_list = [json.loads(el) for el in val_list]
with open(os.path.join(SOURCE_PATH, "test.jsonl"), "r", encoding="utf-8") as file:
    test_list = list(file)
    test_list = [json.loads(el) for el in test_list]

# Create english-only versions of these lists
def english_only(input_list: list) -> list:
    out = []
    for el in input_list:
        if el["annotation_id"].startswith("en_"):
            out.append(el)
    return out
train_list = english_only(train_list)
val_list = english_only(val_list)
test_list = english_only(test_list)

# Replace the old files with the new ones
with open(os.path.join(SOURCE_PATH, "train.jsonl"), "w", encoding="utf-8") as file:
    for element in train_list:
        json.dump(element, file)
        file.write("\n")
with open(os.path.join(SOURCE_PATH, "val.jsonl"), "w", encoding="utf-8") as file:
    for element in val_list:
        json.dump(element, file)
        file.write("\n")
with open(os.path.join(SOURCE_PATH, "test.jsonl"), "w", encoding="utf-8") as file:
    for element in test_list:
        json.dump(element, file)
        file.write("\n")

print("Done!")
