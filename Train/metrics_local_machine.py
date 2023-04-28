import pandas as pd

df = pd.read_parquet(path=r'C:\Users\Noel\Documents\THESIS\Feature Visualization\dataframe_test65')
# print(df)

confusion_accumulator = df.to_numpy()
print(confusion_accumulator)

for i in range(len(confusion_accumulator)):
TP = 
TN = 
FP = 
FN = 