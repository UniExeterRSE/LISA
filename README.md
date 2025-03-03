# Low-Impact Soldier Augmentation Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

![Screen Recording 2024-07-18 at 14 40 42](https://github.com/user-attachments/assets/d33aebfd-4729-41cd-a2ac-e07b48cce7cc)

## Installation
Clone the repository:
```
git clone https://github.com/UniExeterRSE/LISA.git
```

This project uses [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for package management. 

On MacOS, you can install this via:
```
brew install micromamba
```

The environment can then be created and activated:
```
micromamba create -f env.yml
micromamba activate LISA
```
## Tests
To run the tests:
```
pytest
```
## Data
_Under Construction_

To access the data, you will need to set `ONEDRIVE_DIR` in a `.env` file at the root level.

## Project Organisation

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for lisa
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── tests              <- Test files for core functionality.
│   │
│   └── unit           <- Tests for individual functions
│       ├── test_features.py
│       └── test_dataset.py
│
└── lisa                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes lisa a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── modeling         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict.py
    │   └── logistic_regression.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

