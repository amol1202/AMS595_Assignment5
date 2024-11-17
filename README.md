
# AMS595 - Assignment5: Machine Learning Project

This repository contains a Python implementation of various machine learning tasks as part of the AMS595 course. The project covers PageRank, Principal Component Analysis (PCA), Linear Regression, and Gradient Descent optimization. Results are saved and organized in a results folder for easy visualization.


## Table of Contents

- Introduction
- Project Structure
- Tasks Implemented
- How to Run
- Results
- Dependencies
- License
## Introduction

This project demonstrates practical implementations of the following concepts:

- PageRank Algorithm: Simulates the ranking mechanism used by search engines.
- Dimensionality Reduction via PCA: Projects high-dimensional data to a single dimension while preserving maximum variance.
- Linear Regression via Least Squares: Predicts house prices based on features using least-squares regression.
- Gradient Descent: Optimizes a matrix to minimize a mean squared error loss function.
All results are stored in the `results` folder for reproducibility and easy access.
## Project Structure

```
├── machine_learning_with_results.py  # Main Python script
├── results                           # Output folder
│   ├── pagerank_results.txt          # PageRank scores and ranking
│   ├── pca_plot.png                  # PCA plot
│   ├── linear_regression_results.txt # Regression coefficients and predictions
│   ├── gradient_descent_results.txt  # Final loss value after optimization
├── README.md                         # Project documentation (this file)
```

## Run Locally

Clone the project

```bash
  git clone https://github.com/amol1202/AMS595_Assignment5.git
  cd AMS595_Assignment5
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the script

```bash
  python machine_learning_with_results.py
```


## Results

**PageRank scores and rankings**

**Regression coefficients and predicted price**

**Final loss value**

**Visualization of PCA results**

![PCA_Resuls](results/pca_plot.png)

