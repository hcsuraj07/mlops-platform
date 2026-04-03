import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

def download_data():
    print("Downloading UCI Credit Card Default dataset...")

    # fetch dataset — id 350 is "default of credit card clients"
    dataset = fetch_ucirepo(id=350)

    # features and target
    X = dataset.data.features
    y = dataset.data.targets

    # combine into one dataframe
    df = pd.concat([X, y], axis=1)

    # rename columns to meaningful names
    df = df.rename(columns={"Y": "default"})

    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")

    # save raw data
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/credit_default.csv", index=False)
    print("Saved → data/raw/credit_default.csv")

if __name__ == "__main__":
    download_data()