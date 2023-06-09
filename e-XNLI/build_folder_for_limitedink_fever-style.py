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
DESTINATION_DATA_PATH = "../data/e-XNLI-fever/"


# Read .csv files
train_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "train.csv"))
val_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "val.csv"))
test_dataframe = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "test.csv"))


########################################################################################################################
# .JSONL FILES AND DOCS/ FOLDER ########################################################################################
########################################################################################################################
print("Building .jsonl files and the docs/ folder...")
if not os.path.exists(os.path.join(DESTINATION_DATA_PATH, "docs/")):
    os.mkdir(os.path.join(DESTINATION_DATA_PATH, "docs/"))
train_list = []
val_list = []
test_list = []

counts = {
    "ar": 0,  # Arabic
    "bg": 0,  # Bulgarian
    "de": 0,  # German
    "el": 0,  # Greek
    "en": 0,  # English
    "es": 0,  # Spanish
    "fr": 0,  # French
    "hi": 0,  # Hindi
    "ru": 0,  # Russian
    "sw": 0,  # Swahili
    "th": 0,  # Thai
    "tr": 0,  # Turkish
    "ur": 0,  # Urdu
    "vi": 0,  # Vietnamese
    "zh": 0,  # Chinese
}
for dataframe_stage, (source_dataframe, destination_list) in enumerate([(train_dataframe, train_list), (val_dataframe, val_list), (test_dataframe, test_list)]):
    for row in source_dataframe.iterrows():
        # Temp code for debugging
        # if row[1].premise == "Thus, I am assuming that P is an allosteric enhancer of the reaction.":
        #     print("Premise found.")
        language = row[1].language
        label = row[1].label
        # premise_highlighted is already split, so let's use that without the *
        premise = row[1].premise_highlighted.replace('*', '')
        hypothesis = row[1].hypothesis
        premise_highlighted = row[1].premise_highlighted
        hypothesis_highlighted = row[1].hypothesis_highlighted
        # Tokenize and add new file to docs/, putting newlines between sentences.
        # docs_filename = f"{language}_{label}_{str(counts[language][label]).zfill(4)}.txt"
        docs_filename = f"{language}_{str(counts[language]).zfill(4)}.txt"
        # counts[language][label] += 1
        counts[language] += 1
        with open(os.path.join(DESTINATION_DATA_PATH, f"docs/{docs_filename}"), "w", encoding="utf-8") as file:
            file.write(premise)
            # premise_tokenized = ""
            # for word in premise_highlighted.split(" "):
            #     if word.startswith("*") and word.endswith("*"):
            #         premise_tokenized += word[1:-1]
            #     else:
            #         premise_tokenized += word
            #     premise_tokenized += " "
            # premise_tokenized = premise_tokenized[:-1]
            # hypothesis_tokenized = ""
            # for word in hypothesis_highlighted.split(" "):
            #     if word.startswith("*") and word.endswith("*"):
            #         hypothesis_tokenized += word[1:-1]
            #     else:
            #         hypothesis_tokenized += word
            #     hypothesis_tokenized += " "
            # hypothesis_tokenized = hypothesis_tokenized[:-1]
            # to_write_to_this_doc = f"PREMISE :\n{premise_tokenized}\nHYPOTHESIS :\n{hypothesis_tokenized}"
            # file.write(to_write_to_this_doc)
        # Append new element to destination list
        new_element = dict()
        new_element["annotation_id"] = docs_filename
        new_element["classification"] = row[1].label
        new_element["docids"] = [docs_filename]
        evidences = []
        # to_write_to_this_doc_highlighted = f"PREMISE :\n{premise_highlighted}HYPOTHESIS :\n{hypothesis_highlighted}"
        # if to_write_to_this_doc_highlighted.endswith("\n"):
        #     to_write_to_this_doc_highlighted = to_write_to_this_doc_highlighted[:-1]  # Remove trailing newline
        # to_write_to_this_doc_highlighted_split = to_write_to_this_doc_highlighted.replace("\n", " ").split(" ")
        # token_index = 0
        # for token_index, token in enumerate(to_write_to_this_doc_highlighted_split):
        premise_words = premise.split()
        for token_index, token in enumerate(premise_highlighted.split(" ")):
            # TODO: Do we also need to merge runs of contiguous tokens that are all evidence?  Don't know if this matters.
            if token.startswith("*") and token.endswith("*"):
                this_evidence = dict()
                this_evidence["docid"] = docs_filename
                this_evidence["text"] = token[1:-1]
                # this_evidence["start_token"] = token_index
                # TODO : Need to fix evidence spans
                this_evidence["start_token"] = premise_words.index(token[1:-1])
                # this_evidence["start_token"] = 0
                this_evidence["end_token"] = this_evidence["start_token"] + 1
                this_evidence["start_sentence"] = 0
                this_evidence["end_sentence"] = this_evidence["start_sentence"] + 1
                evidences.append([this_evidence])
        new_element["evidences"] = evidences
        new_element["query"] = hypothesis
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


print("Done building the e-XNLI-fever folder!")
