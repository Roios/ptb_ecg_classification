# ECG classification

Electrocardiography (ECG) classification based on the PTB-XL dataset.

## Goal

The goal of this repo is to explore the PTB-XL dataset and try to classify ECGs.

## Data

The data used in this work is from PhysioNet and it is available [here](https://physionet.org/content/ptb-xl/1.0.2/).

You can manually download it or run:

>`wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.2/
`

## Setup

Poetry is used to manage the requirements. Visit [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) for installation details.

>Tip: if you are using VS Code run `poetry config virtualenvs.in-project true` to generate the `.env` in the root

With Poetry installed, one just needs to run `poetry install` from the root of the repo.

## EDA

The first goal is to explore the data. Our data exploration consists on the following steps:

- load the data
- visualize target data
- preprocess the data
  - apply filters
  - estimate the baseline wander
- estimate the QRS complex
- evaluate the data split suggested by `Physionet`
- data augmentation techniques

The EDA is done in `notebooks/eda.ipynb`.
Once we run the notebook, we have a clear understanding on how is the data and how is it split.

## Model

Work in progress.
