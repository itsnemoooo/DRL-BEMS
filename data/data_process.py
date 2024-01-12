# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:09:31 2024

@author: MaxGr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

def randomize(x, factor):
    random_sequence = np.random.uniform(-factor, factor, size=len(x))
    return x + random_sequence    




file_path = 'E:/Data/BMW/BuildingData_2024/building_point_history.csv'
df = pd.read_csv(file_path)
print(df.head())



data_list = df['point_name'].unique()

'''
['DA_FAN_CMD', 'DA_FLOW', 'DAT', 'DA_FLOW_SPT', 'DMPR_POS',
'HTG_STG_01_CMD', 'HTG_CMD', 'HTG_STG_02_CMD', 'OCC_CMD',
'OCC_STS', 'ZN_TEMP', 'ZN_TEMP_CLG_SPT', 'ZN_TEMP_HTG_SPT',
'CHW_VAL_CMD', 'DA_FAN_SPD_CMD', 'DA_SP', 'DAT_SPT', 'ECON_STS',
'MAT', 'OA_DMPR_CMD', 'OA_FLOW', 'RA_DMPR_CMD', 'RA_RH', 'RAT']
'''


df['ts'] = pd.to_datetime(df['ts'])



open_office = True
if open_office is True:
    condition = df['device'].isin(['FPS 2-1','FPS 2-2','FPS 2-3','FPS 2-4','FPS 2-5','FPS 2-6'])
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


time_condition = df['ts'] < datetime(2023, 12, 17)
df = df[time_condition]


condition = (df['point_name'] == 'ZN_TEMP')
ZN_TEMP = df[condition]

groups = ZN_TEMP.groupby('device')
plt.figure(figsize=(10, 6))  # Set the figure size

for name, group in groups:
    plt.plot(group['ts'], group['point_value'], label=name)

plt.title('ZN_TEMP for Different Devices')
plt.xlabel('Time')
plt.ylabel('ZN_TEMP')
plt.legend(loc='best')  # Add a legend
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()







condition = (df['point_name'] == 'HTG_CMD')
HTG_CMD = df[condition]
# HTG_CMD['point_value'] = randomize(HTG_CMD['point_value'])

groups = HTG_CMD.groupby('device')
plt.figure(figsize=(10, 6))  # Set the figure size

for name, group in groups:
    plt.plot(group['ts'], group['point_value'], label=name)

plt.title('HTG_CMD for Different Devices')
plt.xlabel('Time')
plt.ylabel('HTG_CMD')
plt.legend(loc='best')  # Add a legend
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()









condition = (df['point_name'] == 'OCC_STS')
OCC_STS = df[condition]
OCC_STS['point_value'] = randomize(OCC_STS['point_value'], 0.1)

groups = OCC_STS.groupby('device')
plt.figure(figsize=(10, 6))  # Set the figure size

for name, group in groups:
    plt.plot(group['ts'], group['point_value'], label=name)

plt.title('OCC_STS for Different Devices')
plt.xlabel('Time')
plt.ylabel('OCC_STS')
plt.legend(loc='best')  # Add a legend
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()









condition = (df['point_name'] == 'DA_FLOW')
DA_FLOW = df[condition]

groups = DA_FLOW.groupby('device')
plt.figure(figsize=(10, 6))  # Set the figure size

for name, group in groups:
    plt.plot(group['ts'], group['point_value'], label=name)

plt.title('DA_FLOW for Different Devices')
plt.xlabel('Time')
plt.ylabel('DA_FLOW')
plt.legend(loc='best')  # Add a legend
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()






# condition = (df['point_name'] == 'ZN_TEMP_HTG_SPT')
# DA_FLOW = df[condition]

# groups = DA_FLOW.groupby('device')
# plt.figure(figsize=(10, 6))  # Set the figure size

# for name, group in groups:
#     plt.plot(group['ts'], group['point_value'], label=name)

# plt.title('DA_FLOW for Different Devices')
# plt.xlabel('Time')
# plt.ylabel('DA_FLOW')
# plt.legend(loc='best')  # Add a legend
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.show()

























