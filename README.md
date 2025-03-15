# Big Project 1 IF3170 - Machine Learning 2024/2025

## Description

This repository contains the implementation of machine learning algorithms **K-Nearest Neighbors (KNN)**, **Gaussian Naive Bayes**, and **ID3 (Iterative Dichotomiser 3)** applied to the **UNSW-NB15** dataset. The goal of this project is to provide hands-on experience in applying machine learning algorithms to real-world problems and comparing "from scratch" implementations with scikit-learn's library.

## Repository Structure

- `src/` : This folder contains the source code for the algorithm implementations and experiments.
- `doc/` : This folder contains the report in PDF format, including technical explanations and experimental results.
- `README.md` : This document contains guidelines for understanding and running the project.

## Dataset

The **UNSW-NB15** dataset is a collection of network traffic data, including various types of cyberattacks and normal activities. The dataset can be accessed via the following links:
- [Dataset on Kaggle](https://www.kaggle.com/t/ddd18d90f93a47e48f8850b1f1592381)
- [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## Key Features

1. **KNN Implementation (from scratch)**:
   - Supports the number of neighbors as a parameter.
   - Supports 3 distance metrics: Euclidean, Manhattan, and Minkowski.

2. **Gaussian Naive Bayes Implementation (from scratch)**.

3. **ID3 Implementation (from scratch)**:
   - Supports numerical data processing as discussed in the lecture material.

4. **Comparison of Implementation Results with scikit-learn Library**:
   - For ID3, `DecisionTreeClassifier` with the parameter `criterion='entropy'` is used.

5. **Model Persistence**:
   - Models can be saved and loaded in various formats such as `.pkl` or `.txt`.

## Setup

### Prerequisites
- Python 3.8 or newer.
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/jimlynurarif/whitebox.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd whitebox
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Ensure the dataset is prepared in the appropriate directory.
2. Run the notebook to perform experiments:
   ```bash
   jupyter notebook
   ```
3. Select the desired notebook file and follow its instructions.

## Group 

| Nama Anggota         | NIM        | 
|----------------------|------------|
| Jimly Nur Arif       | 13522123   | 
| Samy Muhammad Haikal | 13522151   | 
| Muhammad Roihan      | 13522152   | 

## License

This project is licensed under the [MIT License](LICENSE).

## References

- [The UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [K-Nearest Neighbor (KNN) Algorithm](https://www.geeksforgeeks.org/k-nearest-neighbours/)
- [What Are Naïve Bayes Classifiers?](https://www.ibm.com/topics/naive-bayes)
- [Decision Trees: ID3 Algorithm Explained](https://towardsdatascience.com/decision-trees-for-classification-id3-algorithm-explained-89df76e72df1)
