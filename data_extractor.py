import os
from matplotlib.pyplot import axis
import pandas as pd

directory = os.fsencode('Diabetes-Data')
data_files = []
dfs = []
 
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if "data" in filename:
        data_files.append(filename)

for file in sorted(data_files):
    df = pd.read_csv('Diabetes-Data/'+file, delimiter='	', index_col=None, header=None)
    dfs.append(df)

combined_dfs = pd.concat(dfs, axis=0, ignore_index=True)
combined_dfs.columns = ['Date', 'Time', 'Code', 'Value']
print(combined_dfs)

combined_dfs["Date-time"] = combined_dfs["Date"] + ' ' + combined_dfs["Time"]
combined_dfs.drop(columns=['Date', 'Time'], axis=1, inplace=True)
print(combined_dfs)

# convert to date
combined_dfs['Date-time'] = pd.to_datetime(combined_dfs['Date-time'])
combined_dfs.set_index('Date-time', inplace=True)
combined_dfs.sort_values(by='Date-time', ascending=True)
print(combined_dfs)