"""
Behavioral Debt Spiral — Synthetic Transaction Dataset Generator
================================================================
Generates realistic customer transaction histories with behavioral
distress patterns injected 3–10 months before a default event.

Output files:
  transactions.csv   — one row per transaction (~2M rows for 5K customers)
  customers.csv      — one row per customer with label and metadata
  monthly_summary.csv — one row per customer per month (pre-aggregated)

Usage:
  python generate_dataset.py                    # default 5,000 customers
  python generate_dataset.py --n 50000          # scale up
  python generate_dataset.py --n 1000 --seed 7  # reproducible subset
"""

import argparse
import uuid
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = {
    "essential": [
        "grocery_budget",
        "utility_bill",
        "pharmacy",
        "public_transport",
        "rent_payment",
    ],
    "discretionary": [
        "restaurant",
        "coffee_shop",
        "entertainment",
        "clothing_store",
        "travel_booking",
    ],
    "subscription": [
        "streaming_video",   # cancelled last (most valued)
        "streaming_music",
        "cloud_storage",
        "gym_membership",    # cancelled early (easiest to cut)
        "news_app",
    ],
    "financial": [
        "atm_cash_withdrawal",
        "credit_card_payment",
        "loan_emi",
        "insurance_premium",
    ],
}

# Realistic cancellation order: gym → music → news → cloud → video
SUBSCRIPTION_CANCEL_ORDER = [
    "gym_membership",
    "streaming_music",
    "news_app",
    "cloud_storage",
    "streaming_video",
]

# Average transaction amounts (₹/month) by category group
AMOUNT_PARAMS = {
    "essential":     {"mean": 1_500, "std": 400},
    "discretionary": {"mean": 2_200, "std": 700},
    "subscription":  {"mean":   700, "std": 150},
    "financial":     {"mean": 3_500, "std": 1_200},
}


# ─────────────────────────────────────────────────────────────────────────────
#  DISTRESS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def distress_level(month_idx: int, distress_start: int, severity: float) -> float:
    """
    Returns a 0.0–1.0 float representing how deep into financial distress
    this customer is at this month.  Ramps linearly from distress_start
    to month 18, scaled by severity.
    """
    if month_idx < distress_start:
        return 0.0
    ramp = (month_idx - distress_start) / max(1, 18 - distress_start)
    return min(1.0, ramp * severity)


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOMER GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_customer(n_months: int = 18) -> dict:
    """
    Simulates one customer's full transaction history across n_months.
    Returns a dict with:
      - 'transactions': list of individual transaction dicts
      - 'monthly':      list of per-month summary dicts
      - 'meta':         customer-level metadata dict
    """
    customer_id   = "CUST_" + uuid.uuid4().hex[:8].upper()
    defaulted     = random.random() < 0.25          # 25% default rate
    distress_start = random.randint(3, 10) if defaulted else 999
    severity       = random.uniform(0.6, 1.0) if defaulted else 0.0

    # Financial profile
    annual_income  = random.uniform(240_000, 1_800_000)
    monthly_income = annual_income / 12
    credit_limit   = monthly_income * random.uniform(2, 5)
    base_utilization = random.uniform(0.10, 0.35)

    # Subscriptions all active at start
    active_subs = {s: True for s in SUBSCRIPTION_CANCEL_ORDER}

    txn_records     = []
    monthly_records = []
    start_date      = datetime(2022, 1, 1)

    for m in range(n_months):
        dm         = distress_level(m, distress_start, severity)
        month_date = start_date + timedelta(days=30 * m)

        # ── Income: shrinks and becomes volatile under distress ───────────
        income_noise = random.gauss(0, monthly_income * 0.05 * (1 + dm * 2))
        income = max(0, monthly_income + income_noise - dm * monthly_income * 0.25)

        # ── Credit utilization: climbs under distress ─────────────────────
        utilization = min(
            0.95,
            base_utilization + dm * 0.60 + random.gauss(0, 0.03)
        )
        credit_balance = credit_limit * utilization

        # ── Bill payment delay: 0 days normally → 25+ days at peak distress
        delay_days = max(
            0,
            int(random.gauss(dm * 22, 3 + dm * 8))
        )
        bill_due_date  = month_date + timedelta(days=15)
        bill_paid_date = bill_due_date + timedelta(days=delay_days)

        # ── Subscription cancellations ────────────────────────────────────
        # Cancel one more subscription when distress hits a new threshold
        if dm > 0.15 and random.random() < dm * 0.45:
            for sub in SUBSCRIPTION_CANCEL_ORDER:
                if active_subs[sub]:
                    active_subs[sub] = False
                    break

        n_subs_active = sum(active_subs.values())
        live_subs     = [s for s, v in active_subs.items() if v]

        # ── Spending category mix shifts toward essentials ────────────────
        essential_share    = 0.30 + dm * 0.35
        discretionary_share = max(0.05, 0.35 - dm * 0.28)
        financial_share    = 0.20
        subscription_share = max(0.0, 0.15 - dm * 0.07)

        weights = np.array([
            essential_share,
            discretionary_share,
            financial_share,
            subscription_share,
        ])
        weights /= weights.sum()

        # ── Generate individual transactions ──────────────────────────────
        n_txn = random.randint(10, 35)
        month_spend = 0.0

        for _ in range(n_txn):
            cat_group = random.choices(
                ["essential", "discretionary", "financial", "subscription"],
                weights=weights,
            )[0]

            # For subscriptions, only pick a live one
            if cat_group == "subscription":
                if not live_subs:
                    cat_group = "essential"
                    merchant_cat = random.choice(MERCHANT_CATEGORIES["essential"])
                else:
                    merchant_cat = random.choice(live_subs)
            else:
                merchant_cat = random.choice(MERCHANT_CATEGORIES[cat_group])

            # Amount: discretionary shrinks under distress
            p = AMOUNT_PARAMS[cat_group]
            amount = abs(random.gauss(p["mean"], p["std"]))
            if cat_group == "discretionary":
                amount *= max(0.25, 1.0 - dm * 0.75)

            month_spend += amount
            txn_day = random.randint(1, 28)
            txn_date = month_date + timedelta(days=txn_day)

            txn_records.append({
                "customer_id":       customer_id,
                "date":              txn_date.strftime("%Y-%m-%d"),
                "month_idx":         m,
                "amount":            round(amount, 2),
                "merchant_category": merchant_cat,
                "category_group":    cat_group,
                "defaulted":         int(defaulted),
            })

        # ── Monthly summary row ───────────────────────────────────────────
        monthly_records.append({
            "customer_id":             customer_id,
            "month_idx":               m,
            "year_month":              month_date.strftime("%Y-%m"),
            "income":                  round(income, 2),
            "total_spend":             round(month_spend, 2),
            "credit_balance":          round(credit_balance, 2),
            "credit_limit":            round(credit_limit, 2),
            "utilization_ratio":       round(utilization, 4),
            "bill_payment_delay_days": delay_days,
            "bill_due_date":           bill_due_date.strftime("%Y-%m-%d"),
            "bill_paid_date":          bill_paid_date.strftime("%Y-%m-%d"),
            "n_subs_active":           n_subs_active,
            "essential_ratio":         round(essential_share, 4),
            "distress_level":          round(dm, 4),   # for debugging/validation
            "defaulted":               int(defaulted),
        })

    meta = {
        "customer_id":    customer_id,
        "annual_income":  round(annual_income, 2),
        "credit_limit":   round(credit_limit, 2),
        "distress_start": distress_start if defaulted else None,
        "defaulted":      int(defaulted),
    }

    return {"transactions": txn_records, "monthly": monthly_records, "meta": meta}


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n_customers: int = 5_000, seed: int = 42) -> dict:
    """
    Generates the full dataset and returns three DataFrames:
      transactions_df, monthly_df, customers_df
    """
    np.random.seed(seed)
    random.seed(seed)

    all_txns     = []
    all_monthly  = []
    all_customers = []

    print(f"Generating {n_customers:,} customers × 18 months...")
    for i in range(n_customers):
        if (i + 1) % 1_000 == 0:
            print(f"  {i + 1:,} / {n_customers:,} done")
        result = generate_customer()
        all_txns.extend(result["transactions"])
        all_monthly.extend(result["monthly"])
        all_customers.append(result["meta"])

    txn_df      = pd.DataFrame(all_txns)
    monthly_df  = pd.DataFrame(all_monthly)
    customer_df = pd.DataFrame(all_customers)

    return {
        "transactions": txn_df,
        "monthly":      monthly_df,
        "customers":    customer_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def validate(data: dict) -> None:
    """Prints a quick sanity check to confirm distress signals are realistic."""
    txn  = data["transactions"]
    mon  = data["monthly"]
    cust = data["customers"]

    print("\n" + "═" * 55)
    print("  DATASET VALIDATION REPORT")
    print("═" * 55)

    print(f"\nCustomers total:      {len(cust):>10,}")
    print(f"Default rate:         {cust['defaulted'].mean():>10.1%}")
    print(f"Transaction rows:     {len(txn):>10,}")
    print(f"Monthly summary rows: {len(mon):>10,}")

    d_late  = mon[(mon["defaulted"] == 1) & (mon["distress_level"] > 0.5)]
    d_early = mon[(mon["defaulted"] == 1) & (mon["distress_level"] == 0.0)]
    normal  = mon[mon["defaulted"] == 0]

    print("\n  Payment delay (days)")
    print(f"    Normal customers:        {normal['bill_payment_delay_days'].mean():>6.1f}")
    print(f"    Defaulters (early):      {d_early['bill_payment_delay_days'].mean():>6.1f}")
    print(f"    Defaulters (peak stress):{d_late['bill_payment_delay_days'].mean():>6.1f}")

    print("\n  Credit utilization")
    print(f"    Normal customers:        {normal['utilization_ratio'].mean():>6.1%}")
    print(f"    Defaulters (early):      {d_early['utilization_ratio'].mean():>6.1%}")
    print(f"    Defaulters (peak stress):{d_late['utilization_ratio'].mean():>6.1%}")

    print("\n  Active subscriptions")
    print(f"    Normal customers:        {normal['n_subs_active'].mean():>6.1f}")
    print(f"    Defaulters (peak stress):{d_late['n_subs_active'].mean():>6.1f}")

    print("\n  Merchant category distribution (all transactions)")
    print(txn["category_group"].value_counts(normalize=True).map("{:.1%}".format).to_string())
    print("═" * 55)


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save(data: dict, out_dir: str = ".") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "transactions": out / "transactions.csv",
        "monthly":      out / "monthly_summary.csv",
        "customers":    out / "customers.csv",
    }
    for key, path in paths.items():
        data[key].to_csv(path, index=False)
        print(f"  Saved {key:15s} → {path}  ({len(data[key]):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit transaction data")
    parser.add_argument("--n",    type=int, default=5_000, help="Number of customers")
    parser.add_argument("--seed", type=int, default=42,    help="Random seed")
    parser.add_argument("--out",  type=str, default="data", help="Output directory")
    args = parser.parse_args()

    data = build_dataset(n_customers=args.n, seed=args.seed)
    validate(data)

    print("\nSaving files...")
    save(data, out_dir=args.out)
    print("\nDone. Load with:")
    print("  import pandas as pd")
    print(f"  txn = pd.read_csv('{args.out}/transactions.csv', parse_dates=['date'])")
    print(f"  mon = pd.read_csv('{args.out}/monthly_summary.csv')")
    print(f"  cust = pd.read_csv('{args.out}/customers.csv')")
