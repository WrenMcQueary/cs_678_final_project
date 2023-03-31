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
classes_true = classes_true[fail_indices]
classes_pred = classes_pred[fail_indices]
rationales_pred = rationales_pred[fail_indices, 0, :]
rationales_true = rationales_true[fail_indices, 0, :]
tokens = tokens[fail_indices]


#
# ANALYZE DATA
#
# tokens[4], [11], and [13] are relatively short so could be good for analysis
indices_to_analyze = [4, 11, 13]
classes_true = classes_true[indices_to_analyze]
classes_pred = classes_pred[indices_to_analyze]
rationales_pred = rationales_pred[indices_to_analyze, :]
rationales_true = rationales_true[indices_to_analyze, :]
tokens = tokens[indices_to_analyze]


def get_textprops_and_bracketed_text(r: np.ndarray, t: np.ndarray, p: float) -> tuple:
    """
    Args:
        r: rationale, either true or predicted
        t: tokens for the same phrase as the rationale
        p: Fraction of tokens to highlight.  Set to 0 if ground truth, or a nonzero value if a prediction

    Returns: tuple: (textprops: list, bracketed text: str)
    """
    np.random.seed(0)
    if p:   # If a prediction, threshold as described in the paper:
        # TODO: Apply the Gumbel-softmax trick discussed on page 2, right-hand column of the paper
        all_indices = np.array([ii for ii in range(512)])
        chosen_indices = np.random.choice(all_indices, size=(int(512*p)), replace=False, p=r/np.sum(r))
        binary_rationale_mask = np.array([1 if ii in chosen_indices else 0 for ii in range(min(512, len(t)))])
    else:   # If ground truth:
        binary_rationale_mask = r.copy()
    textprops = [{"bbox": {"edgecolor": "#c9b8e7", "facecolor": "#c9b8e7"}} for _ in range(sum(binary_rationale_mask == 1))]
    bracketed_text = ""
    current_line_length = 0
    for token_counter, token in enumerate(t[:512]):
        if binary_rationale_mask[token_counter] == 1:
            to_add = f"<{token}> "
        else:
            to_add = f"{token} "
        bracketed_text += to_add
        current_line_length += len(token) + 1
        if current_line_length > 90:
            bracketed_text += "\n"
            current_line_length = 0

    return textprops, bracketed_text


for case in range(len(indices_to_analyze)):
    # Build textprops and bracketed text
    textprops_true, bracketed_text_true = get_textprops_and_bracketed_text(rationales_true[case], tokens[case], 0.0)
    textprops_pred, bracketed_text_pred = get_textprops_and_bracketed_text(rationales_pred[case], tokens[case], 0.5)
    bracketed_text_true = f"TRUE CLASS: {classes_true[case]}\n{bracketed_text_true}"
    bracketed_text_pred = f"PREDICTED CLASS: {classes_pred[case]}\n{bracketed_text_pred}"

    # Plot results
    fig, ax = plt.subplots()
    HighlightText(x=0.1, y=0.9,
                  s=bracketed_text_true,
                  fontsize=8,
                  fontname="monospace",
                  highlight_textprops=textprops_true,
                  ax=ax)
    HighlightText(x=0.1, y=0.45,
                  s=bracketed_text_pred,
                  fontsize=8,
                  fontname="monospace",
                  highlight_textprops=textprops_pred,
                  ax=ax)
    plt.show()

print("Done!")
