import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_O_ξ(file_path):
    """Takes file path for best_r.csv files saved out from inference.py"""
    df = pd.read_csv(file_path)
    #df.head()

    xi = df["xi_avg"]
    O_truth = df["O_truth"]
    O_pred = df["O_pred"]

    # get r from filepath string
    filename=Path(file_path).stem
    r_val = filename.split("r")[-1]

    plt.plot(xi, O_truth,label="truth")
    plt.plot(xi,O_pred,label="pred")
    plt.legend()
    plt.title(f"O vs <ξ>, r={r_val}")
    plt.xlabel("<ξ>")
    plt.ylabel("O")
    plt.savefig(f"{filename}.png")
