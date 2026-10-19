"""
AMLWorld (IBM Transactions for AML) audit script -- run this BEFORE building
any graph or training any model, per R4-4's audit protocol.

Checks:
(a) synthetic vs real (documented: AMLWorld is fully synthetic -- confirmed
    in the dataset's own documentation, not something this script can verify
    from the CSV alone, but included here for the record)
(b) label provenance / what "Is Laundering" actually represents
(c) what identifiers exist -- account-level vs institution/bank-level
(d) temporal field availability relative to any prediction window
(e) class prevalence and missingness
(f) whether an institution-level graph is even constructable from this data,
    or whether results should be reported at account level instead

Usage:
    python amlworld_audit.py --trans /workspace/HI-Small_Trans.csv --accounts /workspace/HI-Small_accounts.csv
"""
import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trans", required=True)
    ap.add_argument("--accounts", default=None)
    ap.add_argument("--nrows", type=int, default=None, help="optional row cap for a quick first look")
    args = ap.parse_args()

    print("=" * 70)
    print("STEP 1: Load and inspect transaction file structure")
    print("=" * 70)
    df = pd.read_csv(args.trans, nrows=args.nrows)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

    print("\n" + "=" * 70)
    print("STEP 2: Label provenance and class balance (audit item b, e)")
    print("=" * 70)
    label_col = None
    for c in df.columns:
        if "laundering" in c.lower():
            label_col = c
            break
    if label_col:
        print(f"Label column found: '{label_col}'")
        vc = df[label_col].value_counts()
        print(vc)
        print(f"Positive rate: {df[label_col].mean()*100:.4f}%")
    else:
        print("WARNING: no column with 'laundering' in the name found -- inspect columns manually.")

    print("\n" + "=" * 70)
    print("STEP 3: Identifier granularity -- account vs institution (audit item c)")
    print("=" * 70)
    bank_cols = [c for c in df.columns if "bank" in c.lower()]
    account_cols = [c for c in df.columns if "account" in c.lower()]
    print(f"Bank-related columns: {bank_cols}")
    print(f"Account-related columns: {account_cols}")
    if bank_cols:
        n_banks_from = df[bank_cols[0]].nunique() if len(bank_cols) > 0 else None
        print(f"Unique values in '{bank_cols[0]}': {n_banks_from}")
        if len(bank_cols) > 1:
            n_banks_to = df[bank_cols[1]].nunique()
            print(f"Unique values in '{bank_cols[1]}': {n_banks_to}")
            all_banks = pd.concat([df[bank_cols[0]], df[bank_cols[1]]]).nunique()
            print(f"Union of unique banks across both columns: {all_banks}")
            print(">>> If this number is in the tens/hundreds (not thousands), a bank-to-bank")
            print(">>> graph IS constructable -- this is the institution-level signal we need.")
    if account_cols:
        for c in account_cols:
            print(f"Unique values in '{c}': {df[c].nunique()}")

    print("\n" + "=" * 70)
    print("STEP 4: Temporal field check (audit item d)")
    print("=" * 70)
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "timestamp"])]
    print(f"Time-related columns: {time_cols}")
    if time_cols:
        tc = time_cols[0]
        print(f"Sample values from '{tc}': {df[tc].head(5).tolist()}")
        print(f"Min: {df[tc].min()}, Max: {df[tc].max()}")
        print(">>> Confirm: does the label ('Is Laundering') require knowledge of transactions")
        print(">>> AFTER this timestamp to determine? If patterns span multiple transactions over")
        print(">>> time, a naive split could leak future information -- check the Patterns file.")

    print("\n" + "=" * 70)
    print("STEP 5: Missingness")
    print("=" * 70)
    print(df.isnull().sum())

    if args.accounts:
        print("\n" + "=" * 70)
        print("STEP 6: Accounts file structure")
        print("=" * 70)
        acc = pd.read_csv(args.accounts, nrows=args.nrows)
        print(f"Shape: {acc.shape}")
        print(f"Columns: {list(acc.columns)}")
        print(f"First 3 rows:\n{acc.head(3).to_string()}")
        bank_cols_acc = [c for c in acc.columns if "bank" in c.lower()]
        if bank_cols_acc:
            print(f"\nUnique banks in accounts file ('{bank_cols_acc[0]}'): {acc[bank_cols_acc[0]].nunique()}")

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY -- fill in before proceeding to graph construction")
    print("=" * 70)
    print("(a) Synthetic or real: SYNTHETIC (per IBM/AMLWorld documentation -- confirmed)")
    print(f"(b) Label meaning: '{label_col}' -- confirm against Patterns file for what pattern(s) it flags")
    print("(c) Granularity: see bank/account unique counts above")
    print("(d) Temporal leakage risk: manual check required against Patterns file logic")
    print("(e) Class balance: see Step 2 positive rate")
    print("(f) Recommended prediction level: BANK-level graph if bank union count is small (tens-hundreds);")
    print("    otherwise ACCOUNT-level only -- do not force an institution aggregation without this check.")

if __name__ == "__main__":
    main()
