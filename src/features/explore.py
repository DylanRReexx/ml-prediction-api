import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_PATH = "data/heart.csv"


def explorar():
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("HEART DISEASE DATASET — EXPLORACIÓN")
    print("=" * 50)

    print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

    print("\nColumnas y tipos:")
    print(df.dtypes.to_string())

    print("\nPrimeras 5 filas:")
    print(df.head().to_string())

    print("\nEstadísticas básicas:")
    print(df.describe().round(2).to_string())

    print("\nValores nulos:")
    print(df.isnull().sum().to_string())

    print("\nDistribución del target:")
    print(df["target"].value_counts().to_string())
    balance = df["target"].value_counts(normalize=True).round(3) * 100
    print(f"\n0 (No disease): {balance[0]}%")
    print(f"1 (Disease):    {balance[1]}%")


if __name__ == "__main__":
    explorar()