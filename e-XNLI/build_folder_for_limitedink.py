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
import nltk

nltk.download("punkt")


SOURCE_DATA_PATH = "data/"
DESTINATION_DATA_PATH = "../data/e-XNLI/"


# Read .csv files
train_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "train.csv"))
val_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "val.csv"))
test_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "test.csv"))


########################################################################################################################
# .JSONL FILES AND DOCS/ FOLDER ########################################################################################
########################################################################################################################
print("Building .jsonl files and the docs/ folder...")
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
        # Tokenize and add new file to docs/, putting newlines between sentences.
        docs_filename = f"{language}_{label}_{str(counts[language][label]).zfill(4)}.txt"
        counts[language][label] += 1
        with open(os.path.join(DESTINATION_DATA_PATH, f"docs/{docs_filename}"), "w", encoding="utf-8") as file:
            premise_as_list = nltk.sent_tokenize(premise)
            premise_tokenized = ""
            for sentence in premise_as_list:
                tokens = nltk.word_tokenize(sentence)
                premise_tokenized += " ".join(tokens) + "\n"
            hypothesis_as_list = nltk.sent_tokenize(hypothesis)
            hypothesis_tokenized = ""
            for sentence in hypothesis_as_list:
                tokens = nltk.word_tokenize(sentence)
                hypothesis_tokenized += " ".join(tokens) + "\n"
            to_write_to_this_doc = f"PREMISE :\n{premise_tokenized}HYPOTHESIS :\n{hypothesis_tokenized}"
            if to_write_to_this_doc.endswith("\n"):
                to_write_to_this_doc = to_write_to_this_doc[:-1]    # Remove trailing newline
            file.write(to_write_to_this_doc)
        # Append new element to destination list
        new_element = dict()
        new_element["annotation_id"] = docs_filename
        new_element["classification"] = row[1].label
        evidences = []
        to_write_to_this_doc_highlighted = f"PREMISE :\n{premise_highlighted}HYPOTHESIS :\n{hypothesis_highlighted}"
        if to_write_to_this_doc_highlighted.endswith("\n"):
            to_write_to_this_doc_highlighted = to_write_to_this_doc_highlighted[:-1]  # Remove trailing newline
        to_write_to_this_doc_highlighted_split = to_write_to_this_doc_highlighted.replace("\n", " ").split(" ")
        token_index = 0
        for token_index, token in enumerate(to_write_to_this_doc_highlighted_split):
            # TODO: Do we also need to merge runs of contiguous tokens that are all evidence?  Don't know if this matters.
            if token.startswith("*") and token.endswith("*"):
                this_evidence = dict()
                this_evidence["docid"] = docs_filename
                this_evidence["text"] = token[1:-1]
                this_evidence["start_token"] = token_index
                this_evidence["end_token"] = this_evidence["start_token"] + 1
                if "HYPOTHESIS" in to_write_to_this_doc_highlighted_split[token_index:]:    # We can do this because the word "hypothesis" doesn't appear in the dataset, and because all the samples are only 1 sentence long.     # TODO: Double-check the assumption that all samples are 1 sentence long
                    start_sentence = 1
                else:
                    start_sentence = 3
                this_evidence["start_sentence"] = start_sentence
                this_evidence["end_sentence"] = this_evidence["start_sentence"]
                evidences.append([this_evidence])
        new_element["evidences"] = evidences
        new_element["query"] = "Is the relationship between this premise and this hypothesis entailment, contradiction, or neutral?"
        new_element["query_type"] = None
        destination_list.append(new_element)


# Write .jsonl files
with open(os.path.join(DESTINATION_DATA_PATH, "train.jsonl"), "w", encoding="utf-8") as file:
    for element in train_list:
        json.dump(element, file)
        file.write("\n")
with open(os.path.join(DESTINATION_DATA_PATH, "val.jsonl"), "w", encoding="utf-8") as file:
    for element in val_list:
        json.dump(element, file)
        file.write("\n")
with open(os.path.join(DESTINATION_DATA_PATH, "test.jsonl"), "w", encoding="utf-8") as file:
    for element in test_list:
        json.dump(element, file)
        file.write("\n")


print("Done building the e-XNLI folder!")
