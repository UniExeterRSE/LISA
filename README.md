# Low-Impact Soldier Augmentation Project
The LISA project aims to explore using selected machine learning models to classify a range of human locomotor movements, based on data from wearable sensors. 

Three model types were used to classify movement type, and predict speed and inline of movement: [scikit-learn's](https://github.com/scikit-learn/scikit-learn) logistic/linear regression and random forest, and [microsoft's](https://github.com/microsoft/LightGBM) light gradient boosting machine. <br>
Initial investigations into neural network models are also available in `notebooks/neural_net.ipynb`.

## How to use
Although this repository was developed for a single project, it may be helpful to others in exploring machine learning for biomechanics. The Jupyter notebook at `notebooks/example.ipynb` acts as an interactive tutorial to help guide users on the main workflow. 

`lisa/` contains scripts to transform [C3D](https://www.c3d.org/) data files into [polars](https://github.com/pola-rs/polars) Dataframes, and to extract features relevant to ML training. A framework to then train and evaluate models is also provided.

To view the project output, `models/` contains the pre-trained models and outputs from `multipredictor.py` for the three model types. Each subdirectory contains the pickled model files (plus a scaler for LR) and an `output.json` file detailing the training parameters and validation scores. The RF and LGBM subdirectories also contains json files detailing the feature importances of the models. The confusion matrix and regression histograms produced are also saved.

The *Project Organisation* section below gives a more detailed map of the repository.

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

OS-specific packages should then be installed by running:
```
bash post_install.sh
```
## Tests
To run the tests:
```
pytest
```

## Project Organisation

```
├── data               <- Empty directories for storing your data.
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── lisa               <- Source code for use in this project.
│   │
│   ├── __init__.py    <- Makes lisa a Python module.
│   │
│   ├── config.py      <- Common filepaths and other variables used in other files.
│   │
│   ├── workflow.py    <- Script to run end-to-end process from raw data to trained models.
│   │
│   ├── dataset.py     <- Functions for processing c3d files to a Dataframe,
│   │                     and generating synthetic data.
│   │
│   ├── features.py    <- Functions for extracting required features from the Dataframe.
│   │
│   ├── evaluate.py    <- Functions for evaluating created models.
│   │
│   ├── plots.py       <- Functions for producing evaluation plot.
│   │
│   ├── validation_schema.json   <- Record of previous dataset's column names and types, 
│   │                               that can be used for validating new data.
│   │
│   └── modeling       <- Scripts to train models and then use trained models to make
│       │                 predictions.
│       ├── predict.py             <- Script for applying trained models to new data.
│       ├── multipredictor.py      <- Script for training the classification and 
│       │                             regression models in sequence.
│       └── hyperparameters.json   <- Configuration file for setting model hyperparameters, 
│                                     used in multipredictor.py.
│
├── models             <- Pre-trained models and outputs for each model type.
│
├── notebooks          <- Jupyter notebooks.
│   └── example.ipynb        <- Interactive tutorial on how to use the repository.
│
├── tests              <- Test files for core functionality.
│   │
│   ├── unit           <- Tests for individual functions.
│   │   ├── test_features.py
│   │   └── test_dataset.py
│   └── integration    <- Test for the compete workflow, i.e. raw c3d files to model outputs.
│       └── test_workflow.py
│
├── .gitignore         <- Files/patterns to be ignored by git.
├── .pre-commit-config.yaml   <- Ruff pre-commit code formatter.
├── LICENSE            <- Currently empty.
├── README.md          <- The top-level README for developers using this project.
├── env.yml            <- The micromamba environment, containing the required packages.
└── pyproject.toml     <- Project configuration file with package metadata for lisa
                          and configuration for tools like black.
```

--------

