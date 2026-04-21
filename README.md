🧪 Molecular Solubility Prediction using Machine Learning
🚀 Live Demo

👉 Try the App Here:
🔗 https://hydrasol-ml.streamlit.app/

📌 Overview

This project focuses on predicting the aqueous solubility (LogS) of chemical compounds using machine learning models.

It also includes an interactive Streamlit web app where users can input molecular descriptors and instantly get predictions with insights.

📂 Dataset
Name: delaney_solubility_with_descriptors.csv
The dataset includes:
Molecular descriptors (features)
Experimental solubility values (target)

📊 The goal is to predict LogS (solubility) from these descriptors.

⚙️ Models Used
1. Linear Regression
Baseline model
Assumes linear relationship between features and target
2. Random Forest Regressor
Ensemble-based model
Captures non-linear patterns
Provides better generalization
🧠 Workflow
Data loading & preprocessing
Train-test split
Model training:
Linear Regression
Random Forest
Prediction & evaluation
Visualization of results
Deployment using Streamlit
🌐 Web App Features

✨ Interactive UI with sliders
✨ Real-time prediction updates
✨ Solubility interpretation (Low / Moderate / High)
✨ What-if analysis for feature impact
✨ Clean, modern dashboard design

📊 Evaluation Metrics
Mean Squared Error (MSE): Measures prediction error
R² Score: Measures how well the model explains variance
📈 Visualization

Includes scatter plots comparing:

Actual values (y_test)
Predicted values (y_pred)

Helps evaluate model accuracy visually.

🛠️ Tech Stack
Python 🐍
pandas
numpy
scikit-learn
matplotlib
Streamlit
📁 Project Structure
HydraSol-ML/
│── app.py                  # Streamlit app
│── HydraSol_ML.ipynb      # Model training notebook
│── model_features.pkl     # Feature list
│── solubility_model.pkl   # Trained model
│── requirements.txt       # Dependencies
│── dataset.csv
│── README.md
🚀 How to Run Locally
git clone https://github.com/AgrimaOjha/HydraSol-ML.git
cd HydraSol-ML
pip install -r requirements.txt
streamlit run app.py
💡 Key Learnings
Regression modeling & comparison
Feature-based prediction systems
Model evaluation techniques
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
