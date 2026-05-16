import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
if __name__ == "__main__":
    filepath = "/home/cjr/datasets/Ring2Health/10_second/157a6596.pkl"
    data = pd.read_pickle(filepath)
