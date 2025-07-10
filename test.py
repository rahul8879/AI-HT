
import pandas as pd
df = pd.read_csv("test.csv")
df.Age.value_counts(normalize=True) # age is a col in your data

print('hello')