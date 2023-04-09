"""Use exnli_all_samples.csv to create train.csv, val.csv, and test.csv.
Within each language, the following split will be used: percent_train training, percent_val validation, percent_test testing.
WARNING: This code assumes that each language in the dataset has exactly 5010 samples.
"""


import os
import shutil
import random
import glob as gb
from math import floor
import pandas as pd


samples_per_language = 5010


seed = 0


percent_train = 0.8
percent_val = 0.1
percent_test = 0.1


assert percent_train + percent_val + percent_test == 1


DATA_PATH = "data/"


# Read original as dataframe
original_dataframe = pd.read_csv(os.path.join(DATA_PATH, "exnli_all_samples.csv"))


# Create a separate dataframe for each language, then shuffle each separately.
languages = list(set(original_dataframe["language"].values.tolist()))
languages.sort()
original_dataframe_split_by_language = [original_dataframe.loc[original_dataframe["language"] == language] for language in languages]
for ii, dataframe in enumerate(original_dataframe_split_by_language):
    original_dataframe_split_by_language[ii] = dataframe.sample(frac=1, random_state=seed)


# Combine language dataframes to create train, val, and test dataframes
train_dataframe = pd.DataFrame(columns=["language", "label", "premise", "hypothesis", "premise_highlighted", "hypothesis_highlighted"])
val_dataframe = pd.DataFrame(columns=["language", "label", "premise", "hypothesis", "premise_highlighted", "hypothesis_highlighted"])
test_dataframe = pd.DataFrame(columns=["language", "label", "premise", "hypothesis", "premise_highlighted", "hypothesis_highlighted"])
percents = [percent_train, percent_val, percent_test]
amounts = [round(percent_train * samples_per_language), round(percent_val * samples_per_language), round(percent_test * samples_per_language)]
slicers = [range(0, amounts[0]), range(amounts[0], amounts[0] + amounts[1]), range(amounts[0] + amounts[1], samples_per_language)]
assert sum(amounts) == samples_per_language
for source_dataframe in original_dataframe_split_by_language:
    train_dataframe = train_dataframe.append(source_dataframe.iloc[slicers[0]])
    val_dataframe = val_dataframe.append(source_dataframe.iloc[slicers[1]])
    test_dataframe = test_dataframe.append(source_dataframe.iloc[slicers[2]])


# Write new .csv files
train_dataframe.to_csv(os.path.join(DATA_PATH, "train.csv"), sep=",", index=False)
val_dataframe.to_csv(os.path.join(DATA_PATH, "val.csv"), sep=",", index=False)
test_dataframe.to_csv(os.path.join(DATA_PATH, "test.csv"), sep=",", index=False)


print("Done splitting!")
