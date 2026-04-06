# 🧪 Molecular Solubility Prediction using Machine Learning

## 📌 Overview

This project focuses on predicting the **aqueous solubility (LogS)** of chemical compounds using machine learning models.

The dataset used contains molecular descriptors, which are used as input features to train and evaluate regression models.

---

## 📂 Dataset

* **Name:** `delaney_solubility_with_descriptors.csv`
* The dataset includes:

  * Molecular descriptors (features)
  * Experimental solubility values (target)

📊 The goal is to predict **LogS (solubility)** from these descriptors.

---

## ⚙️ Models Used

### 1. Linear Regression

* A simple baseline model
* Assumes a linear relationship between features and target

### 2. Random Forest Regressor

* Ensemble learning method
* Captures non-linear relationships
* More robust compared to linear models

---

## 🧠 Workflow

1. Load dataset
2. Split into training and testing sets
3. Train models:

   * Linear Regression
   * Random Forest
4. Make predictions
5. Evaluate performance using:

   * Mean Squared Error (MSE)
   * R² Score
6. Visualize results

---

## 📊 Evaluation Metrics

* **Mean Squared Error (MSE):** Measures prediction error
* **R² Score:** Measures how well the model explains variance

---

## 📈 Visualization

The project includes a scatter plot comparing:

* Actual solubility values (y_test)
* Predicted solubility values (y_pred)

This helps in understanding how close predictions are to real values.

---

## 🛠️ Tech Stack

* Python 🐍
* pandas
* numpy
* scikit-learn
* matplotlib

---

## 📁 Project Structure

```
HydraSol-ML/
│── HydraSol_ML.ipynb
│── delaney_solubility_with_descriptors.csv
│── README.md
```

---

## 🚀 How to Run

1. Clone the repository
2. Install required libraries:

   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Run the notebook

---

## 💡 Key Learnings

* Understanding regression models
* Comparing model performance
* Importance of evaluation metrics
* Data visualization for model analysis

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Trying advanced models (XGBoost, Gradient Boosting)
* Feature selection techniques

---

## 👩‍💻 Author

Made with 🩷 by Agrii
