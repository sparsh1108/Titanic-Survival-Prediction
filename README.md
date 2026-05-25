# Titanic Survival Prediction

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning. The goal was to build an end-to-end machine learning pipeline including data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

---

## Dataset

The dataset contains information about Titanic passengers, including:

* Passenger Class (Pclass)
* Sex
* Age
* Fare
* Cabin
* Embarked Port
* Number of Siblings/Spouses
* Number of Parents/Children
* Survival Status (Target Variable)

Target Variable:

* 0 = Did Not Survive
* 1 = Survived

---

## Feature Engineering

The following features were engineered:

### Family Size Feature

A new feature called `with_family` was created:

```python
with_family = SibSp + Parch
```

This feature represents the number of family members traveling with a passenger.

### Cabin Deck Extraction

Only the first letter of the Cabin value was extracted to represent the deck information.

Example:

```text
C85 -> C
B28 -> B
```

---

## Data Preprocessing

The preprocessing pipeline was implemented using Scikit-Learn's Pipeline and ColumnTransformer.

### Categorical Features

* Missing values handled using SimpleImputer
* One Hot Encoding applied using OneHotEncoder
* Unknown categories handled with `handle_unknown='ignore'`

### Numerical Features

* Missing Age values imputed using median strategy
* Features scaled using StandardScaler

---

## Models Used

1. Logistic Regression
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)

---

## Hyperparameter Tuning

GridSearchCV was used to find the best parameters for:

### Random Forest

Parameters tuned:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

### KNN

Parameters tuned:

* n_neighbors
* weights
* distance metric

---

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* YData Profiling

---

## Key Concepts Demonstrated

* Feature Engineering
* Data Cleaning
* Missing Value Handling
* Encoding Categorical Variables
* Feature Scaling
* Pipeline Creation
* ColumnTransformer
* Model Comparison
* Hyperparameter Tuning
* Model Evaluation

---

## Future Improvements

* Add Cross Validation comparison
* Deploy model using FastAPI
* Create interactive prediction API
* Experiment with advanced ensemble methods

---
