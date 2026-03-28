import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("rf_late_delivery_model.pkl")
THRESHOLD = 0.5441219497502292

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Make sure rf_late_delivery_model.pkl is in the repo root."
    )

model = joblib.load(MODEL_PATH)


def score_new_orders(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score new orders for late-delivery risk.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing the same model input features used in training.

    Returns
    -------
    pd.DataFrame
        Original dataframe with:
        - late_delivery_probability
        - late_delivery_prediction
    """
    probs = model.predict_proba(input_df)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    result = input_df.copy()
    result["late_delivery_probability"] = probs
    result["late_delivery_prediction"] = preds
    return result


if __name__ == "__main__":
    sample_input = pd.DataFrame(
        [
            {
                "customer_state": "SP",
                "num_items": 1,
                "total_price": 120.0,
                "total_freight_value": 18.5,
                "num_unique_products": 1,
                "num_unique_sellers": 1,
                "num_payments": 1,
                "total_payment_value": 138.5,
                "total_installments": 1,
                "purchase_month": 11,
                "purchase_dayofweek": 4,
                "purchase_hour": 14,
                "approval_delay_hours": 2.5,
                "estimated_lead_days": 12.0,
            }
        ]
    )

    scored = score_new_orders(sample_input)
    print(scored)