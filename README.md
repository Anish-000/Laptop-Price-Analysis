💻 Laptop Price Prediction using Machine Learning
📌 Project Overview

This project predicts laptop prices based on specifications such as company, type, RAM, storage, processor, GPU, screen features, etc.
The goal is to build a machine learning pipeline that preprocesses the dataset, trains models, and evaluates performance to find the best predictor.

📂 Dataset

Source: laptop_prices.csv (provided dataset)

Rows: 1275

Columns: 23

Target Variable: Price_euros

⚙️ Steps in the Project
1. Data Preprocessing

Checked for missing values → none found.

Converted categorical data into numerical format:

Label Encoding for high-cardinality columns (Product, CPU_model, GPU_model, Screen).

One-Hot Encoding for low-cardinality categorical columns (Company, TypeName, OS, Touchscreen, IPSpanel, RetinaDisplay, CPU_company, GPU_company, PrimaryStorageType, SecondaryStorageType).

2. Feature Selection

Features (X): All columns except Price_euros.

Target (y): Price_euros.

3. Train-Test Split

Split data into 80% training and 20% testing sets.

4. Model Training

Trained two models:

Linear Regression (baseline model)

Random Forest Regressor (advanced model)

5. Model Evaluation

Metrics used:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

6. Visualization

Scatter plots: Actual vs Predicted values.

Regression line plot.

Residual distribution.

Bar chart comparison of model performance (MAE, RMSE, R²).

📊 Results

Linear Regression: Provided a simple baseline but struggled with non-linear relationships.

Random Forest: Outperformed Linear Regression with significantly lower error and higher R² score.

👉 Final conclusion: Random Forest is the better model for laptop price prediction.

🚀 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

📂 Project Structure
├── laptop_prices.csv         # Dataset
├── laptop_price_prediction.ipynb  # Colab Notebook
├── README.md                 # Project description (this file)

📌 Future Improvements

Try Gradient Boosting models (XGBoost, LightGBM).

Perform Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).

Deploy the model using Flask / FastAPI / Streamlit.

✨ Author

👤 Anish Chattopadhyay
📧 [anishchatto2002@gmail.com]
