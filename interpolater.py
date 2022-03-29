import pandas as pd

df = pd.read_csv('bloodsugar_timeseries.csv', parse_dates=[0])
pd.to_numeric(df['Value'])
df.set_index('Date-time', inplace=True)

upsampled = df.resample('30min').mean()
upsampled.loc[upsampled['Code'] == 58, 'Event'] = 'Breakfast'
upsampled.loc[upsampled['Code'] == 60, 'Event'] = 'Lunch'
upsampled.loc[upsampled['Code'] == 62, 'Event'] = 'Dinner'
upsampled.loc[upsampled['Code'] == 64, 'Event'] = 'Snack'

upsampled['Value'] = upsampled['Value'].interpolate(method='linear')
print(upsampled)

upsampled.to_csv('interpolated.csv')