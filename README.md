# Big Project 1 IF3170 - Machine Learning 2024/2025

## Description

This repository contains the implementation of the **Feedforward Neural Network (FFNN)** from scratch, applied to the **mnist_784** dataset.

## Repository Structure

- `src/` : Contains the source code for the algorithm implementations and experiments.
- `doc/` : Contains the project report in PDF format, including technical explanations and experimental results.
- `README.md` : This document provides guidelines for understanding and running the project.

## Dataset
The dataset can be accessed via the following link:
- [mnist_784](https://www.openml.org/search?type=data&sort=runs&id=554&status=active)

## Key Features

1. **FFNN Implementation (from scratch)**:
   - Custom-built FFNN.

2. **Comparison with Scikit-learn Library**:
   - Evaluating FFNN performance against scikit-learn's MLPClassifier.

3. **Visualization of Training Process**:
   - Loss and accuracy plots to analyze model performance.

4. **Model Persistence**:
   - Models can be saved and loaded in `.pkl` format for reuse.

## Setup

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
4. Install Graphviz (required for visualization):
   - Download from [Graphviz official website](https://graphviz.org/download/), or if using a Debian-based Linux system, run:
     ```bash
     sudo apt install graphviz
     ```

## How to Run

1. Ensure the dataset is available in the appropriate directory.
2. Run the Jupyter Notebook for experiments:
   ```bash
   jupyter notebook
   ```
3. Open the desired notebook file and follow the instructions.

## Group Members

| Name                  | NIM        |
|----------------------|------------|
| Jimly Nur Arif       | 13522123   |
| Samy Muhammad Haikal | 13522151   |
| Muhammad Roihan      | 13522152   |

## License

This project is licensed under the [MIT License](LICENSE).

## References

- [MNIST Dataset](https://www.openml.org/search?type=data&sort=runs&id=554&status=active)

