# ml-linear-regression

## Project Overview

This project demonstrates model complexity, overfitting, and underfitting in linear regression. It specifically explores how to balance complexity in polynomial regression models by using higher-order terms and regularization techniques.

### Objectives

1. Obtain a highly complex model (overfitted model) using higher-order terms in polynomial regression.
2. Gradually reduce model complexity using LASSO regularization to get a robust model.
3. Further reduce model complexity to a first-degree polynomial, resulting in an underfitted model.

### Resources

- **Article**: [Linear Regression with Higher-Order Terms](https://data-sorcery.org/2009/06/04/linear-regression-with-higher-order-terms/)
- **Data Set**: [NIST Linear Regression Data - Filip Dataset](https://www.itl.nist.gov/div898/strd/lls/data/Filip.shtml)

---

## Project Structure

- `data/` - Contains the NIST Filip dataset.
- `notebooks/` - Jupyter notebooks demonstrating the steps in model complexity, overfitting, and underfitting.
- `src/` - Python scripts for data loading, model training, and evaluation.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd ml-linear-regression
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the notebook**:
   Use Jupyter Notebook or Jupyter Lab to open and execute the files in `notebooks/`.

---

## Steps to Follow

1. **Create an Overfitted Model**:
   - Use a high-degree polynomial regression (using `sklearn.preprocessing.PolynomialFeatures`).
   - Observe how the model overfits the dataset.

2. **Use LASSO for Regularization**:
   - Apply LASSO regression (using `sklearn.linear_model.Lasso`) to gradually remove the higher-order terms.
   - Monitor the model’s performance and complexity.

3. **Underfitted Model**:
   - Further reduce the polynomial degree to 1 (linear regression).
   - Observe the underfitting and how it affects model performance.

---

## Results and Analysis

Analyze the performance of each model in terms of mean squared error (MSE), complexity, and interpretability. Use plots to visualize the differences between the overfitted, robust, and underfitted models.

---

## License

Add a relevant license if applicable.

## Acknowledgments

- [NIST](https://www.nist.gov/) for the dataset.
- [Data Sorcery](https://data-sorcery.org/) for the linear regression tutorial.
