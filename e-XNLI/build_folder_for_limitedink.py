"""Build a folder in /data/e-XNLI for LimitedInk to use.
Requires e-XNLI/data/test.csv, val.csv, and train.csv to have already been created using create_train_val_test_split.py.
These .jsonl files are formatted so that LimitedInk models can be trained.
"""


import os
import shutil
import random
import glob as gb
from math import floor
import pandas as pd
import json
import re


SOURCE_DATA_PATH = "data/"
DESTINATION_DATA_PATH = "../data/e-XNLI/"


# Read .csv files
train_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "train.csv"))
val_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "val.csv"))
test_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "test.csv"))


########################################################################################################################
# .JSONL FILES AND DOCS/ FOLDER ########################################################################################
########################################################################################################################
if not os.path.exists(os.path.join(DESTINATION_DATA_PATH, "docs/")):
    os.mkdir(os.path.join(DESTINATION_DATA_PATH, "docs/"))
train_list = []
val_list = []
test_list = []
counts = {
    "ar": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "bg": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "de": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "el": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "en": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "es": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "fr": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "hi": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "ru": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "sw": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "th": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "tr": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "ur": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "vi": {"contradiction": 0, "entailment": 0, "neutral": 0,},
    "zh": {"contradiction": 0, "entailment": 0, "neutral": 0,},
}
for dataframe_stage, (source_dataframe, destination_list) in enumerate([(train_dataframe, train_list), (val_dataframe, val_list), (test_dataframe, test_list)]):
    for row in source_dataframe.iterrows():
        language = row[1].language
        label = row[1].label
        premise = row[1].premise
        hypothesis = row[1].hypothesis
        premise_highlighted = row[1].premise_highlighted
        hypothesis_highlighted = row[1].hypothesis_highlighted
        # Add new file to docs/
        docs_filename = f"{language}_{label}_{str(counts[language][label]).zfill(4)}.txt"
        counts[language][label] += 1
        with open(os.path.join(DESTINATION_DATA_PATH, f"docs/{docs_filename}"), "w", encoding="utf-8") as file:
            file.write(f"PREMISE : {premise}\nHYPOTHESIS : {hypothesis}")
        # Append new element to destination list
        new_element = dict()
        new_element["annotation_id"] = docs_filename
        new_element["classification"] = row[1].label
        # Build evidences
        evidences_list = []
        matches = re.finditer(r'\*[^ ]+\*', premise_highlighted)     # Find 2 consecutive asterisks
        for match in matches:
            single_evidence = {}
            single_evidence['docid'] = docs_filename
            single_evidence['start_token'] = match.start()
            single_evidence['end_token'] = match.end()
            single_evidence['text'] = match.group()
            evidences_list.append(single_evidence)
        new_element["evidences"] = evidences_list     # TODO
        new_element["query"] = "Is the relationship between this premise and this hypothesis entailment, contradiction, or neutral?"
        new_element["query_type"] = None
        destination_list.append(new_element)


# Write .jsonl files
# TODO


########################################################################################################################
# HUMAN_ANNOTATIONS.PICKLE #############################################################################################
########################################################################################################################
# TODO


print("Done building folder!")
