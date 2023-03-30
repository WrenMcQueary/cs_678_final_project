"""For our baseline report.
Obtain some instances where the model fails on the Movies dataset.
"""


import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from highlight_text import HighlightText, ax_text, fig_text


#
# OBTAIN DATA
#
# Obtain guessed rationales
with np.load("../../checkpoints/movies/distilbert/token_rationale/length_level_0.5/seed_1234/rationale.npz", allow_pickle=True) as file:
    rationales_pred = file["data"]
    classes_true = file["true"]
    classes_pred = file["pred"]

# Obtain ground-truth rationales and tokens
with open("../../data/movies/human_annotations.pickle", "rb") as file:
    data = pickle.load(file)
rationales_true = data["evidence_masks"]
tokens = data["tokens"]

# Unbatch guessed rationales so the shape is the same as the ground-truth rationales
rationales_pred_flattened = np.zeros((199, 1, 512))
sample_counter = 0
for slice_counter in range(67):
    for sliver_counter in range(3):
        if sliver_counter >= np.shape(rationales_pred[slice_counter])[0]:
            break
        rationales_pred_flattened[sample_counter, 0, :] = rationales_pred[slice_counter][sliver_counter, :]
        sample_counter += 1
rationales_pred = rationales_pred_flattened.copy()
del rationales_pred_flattened

# Filter for cases where the model guessed the wrong class
fail_indices = []
for ii in range(199):
    if classes_true[ii] != classes_pred[ii]:
        fail_indices.append(ii)
rationales_pred = rationales_pred[fail_indices, 0, :]
rationales_true = rationales_true[fail_indices, 0, :]
tokens = tokens[fail_indices]


#
# ANALYZE DATA
#
# Break so that these can be analyzed.  tokens[4], [11], and [13] are relatively short so could be good for analysis
# TODO: Plot a comparison of true to pred.  Will have to threshold true.  See paper for this; it might already be thresholding.
indices_to_analyze = [4, 11, 13]
rationales_pred = rationales_pred[indices_to_analyze, :]
rationales_true = rationales_true[indices_to_analyze, :]
tokens = tokens[indices_to_analyze]

for case in range(len(indices_to_analyze)):
    # Build textprops
    true_textprops = [{"bbox": {"edgecolor": "#c9b8e7", "facecolor": "#c9b8e7"}} for ii in range(sum(rationales_true[case, :]))]
    breakpoint()    # TODO: Remove debugging line
    #pred_textprops = [{"bbox": {"edgecolor": "#c9b8e7", "facecolor": "#c9b8e7"}} for ii in range(sum(rationales_true[case, :]))]
    text_with_angle_brackets_true = ""
    current_line_length = 0
    for token_counter, token in enumerate(tokens[case][:512]):
        if rationales_true[case, token_counter] == 1:
            to_add = f"<{token}> "
        else:
            to_add = f"{token} "
        text_with_angle_brackets_true += to_add
        current_line_length += len(token) + 1
        if current_line_length > 120:
            text_with_angle_brackets_true += "\n"
            current_line_length = 0

    # Plot results
    fig, ax = plt.subplots()
    HighlightText(x=0.1, y=0.9,
                  s=text_with_angle_brackets_true,
                  fontsize=8,
                  fontname="monospace",
                  highlight_textprops=true_textprops,
                  ax=ax)
    plt.show()

breakpoint()    # TODO: Remove debugging line
print("Done!")
