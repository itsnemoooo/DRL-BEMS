# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 08:25:06 2024

@author: MaxGr
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

def randomize(x, factor):
    random_sequence = np.random.uniform(-factor, factor, size=len(x))
    return x + random_sequence    




file_path = 'E:/Data/BMW/BuildingData_2024/new_hvac_status_history.csv'
df = pd.read_csv(file_path)
print(df.head())



data_list = df['vav'].unique()

'''
['VAV-01-201', 'VAV-01-202', 'VAV-01-203', 'VAV-01-204',
'VAV-01-205', 'VAV-01-206', 'FPS 2-1', 'FPS 2-2', 'FPS 2-3',
'FPS 2-4', 'FPS 2-5', 'FPS 2-6', 'FPS 2-17', 'FPS 2-18',
'FPS 2-19', 'FPS 2-20', 'FPS 2-21', 'FPS 2-10', 'FPS 2-11',
'FPS 2-12', 'FPS 2-13', 'FPS 2-14']
'''


df['ts'] = pd.to_datetime(df['ts'])



open_office = True
if open_office is True:
    condition = df['vav'].isin(['FPS 2-1','FPS 2-2','FPS 2-3','FPS 2-4','FPS 2-5','FPS 2-6'])
    # condition = df['device'].isin([
    #     # 'FPS 2-7','FPS 2-8','FPS 2-9','FPS 2-10',
    #                                # 'FPS 2-11','FPS 2-12',
    #                                'FPS 2-13','FPS 2-14',
    #                                'FPS 2-15','FPS 2-16','FPS 2-17','FPS 2-18',
    #                                'FPS 2-19','FPS 2-20','FPS 2-21'])

    df = df[condition]




time_shift = True
if time_shift is True:
    df['ts'] = df['ts'] - pd.Timedelta(hours=5)


# time_condition = df['ts'] < datetime(2023, 12, 17)
# df = df[time_condition]



# condition = (df['point_name'] == 'ZN_TEMP')
# ZN_TEMP = df[condition]

df['occupied'] = randomize(df['occupied'], 0.1)

groups = df.groupby('vav')
plt.figure(figsize=(10, 6))  # Set the figure size

for name, group in groups:
    plt.plot(group['ts'], group['occupied'], label=name)

plt.title('OCC for Different Devices')
plt.xlabel('Time')
plt.ylabel('OCC')
plt.legend(loc='best')  # Add a legend
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()











