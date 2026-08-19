# KNN Machine Learning – Synthetic Dataset

## 📌 Project Overview

This project creates a **synthetic classification dataset** using `make_blobs()` from Scikit-learn and performs basic **data exploration and visualization** using Python.

The dataset contains **4000 samples**, **3 features**, and **3 classes**. The project also prepares the data for applying the **K-Nearest Neighbors (KNN)** classification algorithm.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📦 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### Purpose of Libraries

| Library      | Purpose                          |
| ------------ | -------------------------------- |
| NumPy        | Numerical operations and arrays  |
| Pandas       | Data manipulation and DataFrames |
| Matplotlib   | Data visualization               |
| Seaborn      | Statistical visualization        |
| Scikit-learn | Machine learning tools           |

---

## 📊 Dataset

The dataset is generated using:

```python
X, y = make_blobs(
    n_samples=4000,
    centers=3,
    n_features=3,
    random_state=80,
    cluster_std=2
)
```

### Dataset Parameters

| Parameter      | Value | Meaning                                  |
| -------------- | ----: | ---------------------------------------- |
| `n_samples`    |  4000 | Total number of observations             |
| `centers`      |     3 | Number of classes/clusters               |
| `n_features`   |     3 | Number of input features                 |
| `random_state` |    80 | Makes the generated dataset reproducible |
| `cluster_std`  |     2 | Controls the spread of the clusters      |

### Features and Target

* `X` → Input features
* `y` → Target/class labels

The shape of `X` is:

```text
(4000, 3)
```

The shape of `y` is:

```text
(4000,)
```

---

## 📈 Data Visualization

### Scatter Plot

The project visualizes the first two features:

```python
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    marker='.',
    s=10,
    edgecolors='blue'
)
```

Here:

* `X[:, 0]` → First feature
* `X[:, 1]` → Second feature
* `c=y` → Colors the points according to their class
* `marker='.'` → Uses dots as markers
* `s=10` → Controls marker size

This allows us to visually observe the three generated classes.

---

## 📋 DataFrame

The feature data is converted into a Pandas DataFrame:

```python
df = pd.DataFrame(X)
```

The first five records can be viewed using:

```python
df.head()
```

The DataFrame contains three columns representing the three features.

---

## 📊 Histogram

Histograms are created to understand the distribution of the features:

```python
df.plot(
    kind='hist',
    subplots=True,
    layout=(5, 2),
    sharex=False,
    sharey=False,
    bins=100
)
```

The histograms help identify how the values of each feature are distributed.

---

## 🤖 KNN Preparation

The project imports the following KNN-related tools:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

These will be used to:

1. Split the dataset into training and testing data.
2. Create a KNN classifier.
3. Train the classifier.
4. Make predictions.
5. Calculate the model's accuracy.

### Important

The current code **does not yet train the KNN model**. It currently focuses on:

```text
Data Generation
       ↓
Data Inspection
       ↓
Data Visualization
       ↓
DataFrame Creation
       ↓
Feature Distribution
```

The KNN classification stage can be added next.

---

## 🚀 How to Run

### 1. Install Python

Make sure Python is installed on your computer.

### 2. Install Required Libraries

Run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run the Python File

Save the code as:

```text
knn.py
```

Then run:

```bash
python knn.py
```

---

## 📁 Project Structure

```text
KNN-Project/
│
├── knn.py
└── README.md
```

---

## 🎯 Learning Objectives

This project helps understand:

* Synthetic dataset generation
* Features and target variables
* NumPy arrays
* Pandas DataFrames
* Dataset shapes
* Scatter plots
* Histograms
* Basic Scikit-learn workflow
* Preparation for KNN classification

---

## 🔮 Future Improvements

The project can be extended by adding:

* Train-test split
* KNN model creation
* Different values of `K`
* Model prediction
* Accuracy calculation
* Confusion matrix
* Classification report
* Comparison of different K values
* Decision boundary visualization

---

## 👩‍💻 Author

**Sameer **

BTECH Student

---

## 📄 License

This project is created for **educational and learning purposes**.
