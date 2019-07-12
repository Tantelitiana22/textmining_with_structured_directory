TextMiningPourFileRouge
==============================

This project aim to predict job offer label, and to understand how to use several approach in NLP field.
To project our corpus in a numerical dimension, one test some word embedding method like:
    - TF-IDF 
    - Word2Vec
    - fastText
    - Glove
   Afert the projection of our corpus in a numerical dimension, one applied with the matrix projected a machine learning modeles
   like:
   - Logistic regression
   - Linear SVM 
   - Naive Bayes
   - Gradient Boosting
   - Random Forest.
One also tested deep learning models by using a sequential method as word embedding. A specialy:
   - Neural Network with 2 layers.
   - Convolutional Neural Network.
   - LTSM.
  For all word embedding+models we tested, the best is TF-IDF+Logistic regression. After choising the best model, one use it 
  in a little and simple application  we created with python. The biggest default of this application the fact that it only
  work with a very specific database.  To lauch the application, you have first to install the requirements package by doing:
  ```python
pip install -r requirements.txt
print s
```
After you can lauch the file named  upload_file_Application.py by doing the next command line:

 ```python
python  upload_file_Application.py
```
A window will open, and to test the application, you can upload data in jeu_test file named UNLabeledTestdata.csv to test. 
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
