"""To be called with a train.log file as a target.  Generates a plot of loss curves.
Example usage:
python plot_loss_curves.py -f checkpoints/exnli/xlm-r/token_rationale/length_level_0.5/seed_1234/train.log
"""


import argparse
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", action="store", nargs=1, type=str, required=True, help="Filepath of the the train.log file", dest="filepath")
    args = parser.parse_args()
    filepath = args.filepath[0]

    with open(filepath, "r", encoding="utf-8") as file:
        file_content_as_string = file.read()
        file_content = file_content_as_string.split("\n")

    # Parse training losses and validation losses
    losses_train = [line for line in file_content if "| Train Loss: " in line]
    for ll, line in enumerate(losses_train):
        losses_train[ll] = float(line.split(" ")[-2])
    losses_val = [line for line in file_content if "| Validation Loss: " in line]
    for ll, line in enumerate(losses_val):
        losses_val[ll] = float(line.split(" ")[-2])
    epoch_vector = [ii for ii in range(len(losses_train))]

    # Trim lists
    if "============= LimitedInk Model Evaluating =============" in file_content_as_string:
        losses_val = losses_val[:-1]

    # Plot
    plt.plot(epoch_vector, losses_train, label="train loss")
    plt.plot(epoch_vector, losses_val, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curves")
    plt.legend()
    plt.show()
