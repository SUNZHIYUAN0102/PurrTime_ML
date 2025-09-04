import pandas as pd

df = pd.read_csv("collar_data6.csv")

# 只保留    'X_Mean','X_Min','X_Max','X_Sum',
#     'Y_Mean','Y_Min','Y_Max','Y_Sum',
#     'Z_Mean','Z_Min','Z_Max','Z_Sum'

df = df[['Cat_id', 'Timestamp', 
         'X_Mean','X_Min','X_Max','X_Sum',
         'Y_Mean','Y_Min','Y_Max','Y_Sum',
         'Z_Mean','Z_Min','Z_Max','Z_Sum','Behaviour']]

df.to_csv("harness.csv", index=False)