import pandas as pd

train = pd.read_csv("dataset/emnist-letters-train.csv")

with open("metrics.txt", "w") as f:
    f.write(str(len(train.columns)))

