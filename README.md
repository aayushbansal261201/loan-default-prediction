# Loan Default Prediction System

## Overview

This project focuses on predicting whether a customer is likely to default on a loan. The goal is to help in identifying high-risk customers so better lending decisions can be made.

## Objective

To build a machine learning model that can detect potential loan defaulters, with more importance given to catching defaulters rather than minimizing false alarms.

## Tech Stack

* Python
* Pandas, NumPy, Scikit-learn
* SMOTE (for handling imbalanced data)
* Streamlit (for deployment)

## Approach

* Performed data cleaning and preprocessing
* Handled class imbalance using SMOTE
* Trained a Logistic Regression model
* Focused on improving recall since missing a defaulter is more costly

## How to Run

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

## Project Structure

```
app/
  app.py
data/
notebooks/
model/
requirements.txt
README.md
```

## Future Improvements

* Try advanced models like Random Forest or XGBoost
* Add better visualization for insights
* Improve the UI of the Streamlit app
* Deploy the project online

## Note

This project is built for learning and demonstration purposes but follows a practical approach used in real-world risk modeling.
