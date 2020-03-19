import pandas as pd
import numpy as np

data = pd.read_csv("student_data.csv")

one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
print(one_hot_data[:10])

#one_hot_data = data.drop(columns=["rank"])

#print(one_hot_data[:10])