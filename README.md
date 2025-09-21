# 🏡 House Price Prediction using XGBoost

💰 This project predicts **house prices** using the **Boston Housing Dataset** and **XGBoost Regressor**.

---

## ⚙️ Workflow

1️⃣ **House Price Data** 📊 – Load dataset with 506 rows & 14 columns.
2️⃣ **Data Preprocessing** 🧹 – Explore correlations and clean the data.
3️⃣ **Train-Test Split** ✂️ – Split into training and testing sets.
4️⃣ **XGBoost Regressor Model** 🧠 – Train the model to learn price patterns.
5️⃣ **Prediction** 🔮 – Feed new data → predict house price.

---

## 📦 Importing Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```

---

## 📊 Dataset Overview

```python
# number of rows and columns
house_price_dataframe.shape
# Output: (506, 14)
```

---

## 🔎 Correlation Analysis

```python
correlation = house_price_dataframe.corr()

# constructing a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size':8}, cmap='Blues')
```

---

## 🎯 Model Performance

### ✅ Training Data

```python
# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)
# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

# Output:
# R squared error :  0.9733
# Mean Absolute Error :  1.1453
```

### ✅ Test Data

```python
test_data_prediction = model.predict(X_test)

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)
# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

# Output:
# R squared error :  0.9116
# Mean Absolute Error :  1.9922
```

---

## 🛠️ How to Run this Project

### 🔹 Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Run the Code

```bash
python main.py
```

---

## ⭐ Support

If you found this repo useful, **don’t forget to star ⭐ the repository!** 🚀

---
