# Olist Late Delivery Prediction

## Project Overview
This project predicts whether an e-commerce order will be delivered late using the Olist Brazilian E-Commerce dataset. The work was completed as part of a Business Analytics course project and includes data preprocessing, exploratory data analysis, predictive modeling, model explainability, bias analysis, and a Power BI dashboard.

## Business Problem
Late deliveries reduce customer satisfaction, damage seller reputation, and create operational inefficiencies in e-commerce. The goal of this project is to identify the main factors associated with delayed deliveries and build a machine learning model that predicts orders at risk of being delivered late.

## Dataset
The project uses the Olist Brazilian E-Commerce dataset, which contains multiple related tables covering:
- orders
- order items
- order payments
- order reviews
- products
- customers
- sellers
- geolocation

Target variable:
- `late_delivery = 1` if the order was delivered after the estimated delivery date
- `late_delivery = 0` otherwise

## Project Workflow
The notebook covers:
1. Data loading and table merging
2. Data cleaning and preprocessing
3. Target creation
4. Exploratory data analysis
5. Feature engineering
6. Model training and comparison
7. Final model selection
8. Explainability using SHAP and permutation importance
9. Bias analysis by state
10. Dashboard export preparation

## Final Model
Selected final model:
- **Tuned Random Forest**

Final model metrics:
- ROC AUC: **0.7611**
- Precision (late class): **0.27**
- Recall (late class): **0.46**
- F1 Score (late class): **0.34**

## Key Predictors
The most influential predictors of late delivery were:
- purchase_month
- estimated_lead_days
- customer_state
- total_freight_value
- approval_delay_hours

## Repository Contents
- `late_delivery_prediction.ipynb` → full project notebook
- `score_new_orders.py` → scoring script for deployment-style prediction
- `requirements.txt` → Python dependencies
- `rf_late_delivery_model.pkl` → saved trained model
- `images/` → project figures used in GitHub and presentation

## How to Run
1. Install dependencies:
   `pip install -r requirements.txt`

2. Open the notebook:
   `jupyter notebook late_delivery_prediction.ipynb`

3. Run the scoring script after the model file is available:
   `python score_new_orders.py`

## Dashboard and Presentation
This project also includes:
- a Power BI dashboard
- a PowerPoint presentation

These are submitted separately through the course submission system.
