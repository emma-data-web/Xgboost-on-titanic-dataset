# Xgboost-on-titanic-dataset
This project builds a machine learning model to predict housing prices in the USA using the [USA Housing Dataset](https://www.kaggle.com/datasets/saurabhshahane/usa-housing). It uses a **scikit-learn pipeline** for clean preprocessing, **XGBoost** for regression modeling, and **GridSearchCV** for hyperparameter tuning.

---

## 📦 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

---

## 🔍 Project Highlights

- 📁 Cleaned and preprocessed real estate data
- 🔁 Used a `Pipeline()` from scikit-learn for consistent preprocessing and modeling
- ⚙️ Tuned hyperparameters using `GridSearchCV`
- 🎯 Evaluated using MAE, RMSE, and R²
- 📊 Visualized **feature importances** to interpret the model

---

## 📊 Dataset Overview

The dataset includes:
- `Avg. Area Income`
- `Avg. Area House Age`
- `Avg. Area Number of Rooms`
- `Avg. Area Number of Bedrooms`
- `Area Population`

**Target:**
- `Price`

---

## ⚙️ Model Workflow

1. **Data Cleaning**  
   - Removed irrelevant or redundant features if needed.
  
2. **Feature Scaling**  
   - Scaled numerical data using `StandardScaler` within the pipeline.

3. **Modeling**  
   - Used `XGBRegressor()` as the core estimator inside the pipeline.

4. **Hyperparameter Tuning**  
   - Used `GridSearchCV` to find the best `n_estimators`, `learning_rate`, `max_depth`, etc.

5. **Feature Importance**  
   - Plotted the top features driving price prediction using `model.feature_importances_`.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/usa-housing-xgboost.git
cd usa-housing-xgboost
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the script or notebook
bash
Copy
Edit
python housing_pipeline.py
or open and run the notebook:

bash
Copy
Edit
jupyter notebook usa_housing_xgboost.ipynb

👤 Author
Made with ❤️ by Nwankwo Emmanuel Ota
