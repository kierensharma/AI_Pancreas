from math import comb
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

combined_dfs["Date-time"] = combined_dfs["Date"] + ' ' + combined_dfs["Time"]
combined_dfs.drop(columns=['Date', 'Time'], axis=1, inplace=True)

# convert to date
combined_dfs['Date-time'] = pd.to_datetime(combined_dfs['Date-time'])
combined_dfs = combined_dfs.dropna()
combined_dfs.set_index('Date-time', inplace=True)
combined_dfs.sort_values(by='Date-time', ascending=True, inplace=True)

# combined_dfs.to_csv('concat_data.csv')

# bloodsugar_codes = ['48','57','58','59','60','61','62','63','64']
bloodsugar_df = combined_dfs.loc[(combined_dfs['Code']>=48) & (combined_dfs['Code']<=64)]
bloodsugar_df.loc[bloodsugar_df['Code'] == 58, 'Event'] = 'Breakfast'
bloodsugar_df.loc[bloodsugar_df['Code'] == 60, 'Event'] = 'Lunch'
bloodsugar_df.loc[bloodsugar_df['Code'] == 62, 'Event'] = 'Dinner'
bloodsugar_df.loc[bloodsugar_df['Code'] == 64, 'Event'] = 'Snack'

print(bloodsugar_df)
# bloodsugar_df.to_csv('bloodsugar_timeseries.csv')