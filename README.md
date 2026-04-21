# 🧪 Molecular Solubility Prediction using Machine Learning

## 🚀 Live Demo

👉 **Try the App Here:**  
🔗 https://hydrasol-ml.streamlit.app/

---

## 📌 Overview

This project focuses on predicting the **aqueous solubility (LogS)** of chemical compounds using machine learning models.

It also includes an **interactive Streamlit web app** where users can input molecular descriptors and instantly get predictions with insights.

---

## 📂 Dataset

- **Name:** `delaney_solubility_with_descriptors.csv`

The dataset includes:
- Molecular descriptors (**features**)
- Experimental solubility values (**target**)

📊 The goal is to predict **LogS (solubility)** from these descriptors.

---

## ⚙️ Models Used

### 🔹 Linear Regression
- Simple baseline model  
- Assumes linear relationship between features and target  

### 🌳 Random Forest Regressor
- Ensemble learning method  
- Captures non-linear relationships  
- More robust and accurate  

---

## 🧠 Workflow

1. **Load dataset**  
2. **Split into training and testing sets**  
3. **Train models**
   - Linear Regression  
   - Random Forest  
4. **Make predictions**  
5. **Evaluate performance**
   - Mean Squared Error (MSE)  
   - R² Score  
6. **Visualize results**  
7. **Deploy using Streamlit**

---

## 🌐 Web App Features

✨ **Interactive UI with sliders**  
✨ **Real-time prediction updates**  
✨ **Solubility interpretation (Low / Moderate / High)**  
✨ **What-if analysis for feature impact**  
✨ **Modern dashboard design**

---

## 📊 Evaluation Metrics

- **Mean Squared Error (MSE):** Measures prediction error  
- **R² Score:** Measures how well the model explains variance  

---

## 📈 Visualization

The project includes scatter plots comparing:

- **Actual values (y_test)**  
- **Predicted values (y_pred)**  

Helps evaluate model performance visually.

---

## 🛠️ Tech Stack

- **Python 🐍**  
- **pandas**  
- **numpy**  
- **scikit-learn**  
- **matplotlib**  
- **Streamlit**

---

## 📁 Project Structure
HydraSol-ML/
│── app.py # Streamlit app
│── HydraSol_ML.ipynb # Model training notebook
│── model_features.pkl # Feature list
│── solubility_model.pkl # Trained model
│── requirements.txt # Dependencies
│── delaney_solubility_with_descriptors.csv
│── README.md

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/AgrimaOjha/HydraSol-ML.git
cd HydraSol-ML
pip install -r requirements.txt
streamlit run app.py

💡 Key Learnings
Understanding regression models
Comparing model performance
Importance of evaluation metrics
Building interactive ML apps
Deployment using Streamlit Cloud
🔮 Future Improvements
Hyperparameter tuning
Cross-validation
Advanced models (XGBoost, Gradient Boosting)
Feature importance visualization
API integration
👩‍💻 Author

Made with 🩷 by Agrii
