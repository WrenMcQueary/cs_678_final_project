Final project for CS 678 at George Mason University.  Builds on top of the tool [LimitedInk](https://github.com/huashen218/LimitedInk) and the dataset [explaiNLI (aka e-XNLI)](https://github.com/KeremZaman/explaiNLI).

## Data
The data for our multilinguality experiments is contained in the e-XNLI directory. It contains the original e-XNLI csv file as well as split train, test, and val csv files. The \data\e-XNLI-fever directory needs to be populated with the properly formatted data in order to run the e-XNLI experiments. To populate the directory, run:
```
$ ./build_exnli_fever_data.sh
```
Passing an -e option also builds the \data\e-XNLI-just-english directory, which may be useful for testing and debugging purposes. This can be executed as follows:
```
$ ./build_exnli_fever_data.sh -e
