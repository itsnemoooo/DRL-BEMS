#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:20:00 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import time
import copy
import math
import shutil
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable


import yaml

# Define the parameters
config = {
    'openstudio_path': './openstudioapplication-1.6.0/',
    'EPlus_file': './openstudioapplication-1.6.0/EnergyPlus',
    'osm_name_box': './building_model/ITRC_2nd_6zone_OPEN_3.60.osm',
    'weather_data': './weather_data/USA_SC_Greenville-Spartanburg.Intl.AP.723120_TMY3.epw',
    'iddfile': 'Energy+.idd',
    'save_idf': 'run.idf',
    'weight_file': './weights/last.pth',
    'replay_buffer': 'replay_buffer.pkl',
    
    'HVAC_output': True,
    'reset_dataframe': False,
    'Train': True,
    'signal_loss': False,
    'AirWall_Switch': 'on',
    'Roof_Switch': 'off',
    
    'timestep_per_hour': 12,
    'begin_month': 1,
    'begin_day_of_month': 1,
    'end_month': 12,
    'end_day_of_month': 31,

    
    'state_dim': 14,
    'action_dim': 64,
    'epochs': 10,
    'lr': 0.001,
    'gamma': 0.9,
    'epsilon': 0,
    
    'target_update': 10,
    'buffer_size': 1000,
    'minimal_size': 200,
    'batch_size': 128,
    'FPS': 1000,
    
    'positive_reward': 0,
    'signal_factor': 0,
    'T_factor_day': 1,
    'E_factor_day': 1e-5,
    'T_factor_night': 0.1,
    'E_factor_night': 1e-5,
    
    'F_bottom': 60,
    'F_top': 90
}


output_file = 'config.yaml'

# Save parameters to a YAML file
with open(output_file, 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)

print(f'Parameters saved to {output_file}')



class Parameters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
config = Parameters(**config)






# Specify the folder paths you want to create
folder_paths = [
    "Benchmark_data",
    "data",
    "weights",
    "plot"
]

for folder_path in folder_paths:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")



def save_parameters_to_txt(parameters, file_path):
    with open(file_path, 'w') as file:
        for key, value in parameters.items():
            file.write(f'{key}: {value}\n')
            print(f"{key}: {value}")
    print(f'Parameters saved to {file_path}')

    






import datetime
import pandas as pd

ts = datetime.datetime.now()
ts = datetime.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)

History = {
    'time_line': [ts],
    'year': [ts.year],
    'month': [ts.month],
    'day': [ts.day],
    'hour': [ts.hour],
    'minute': [ts.minute],
    
    'weekday': [ts.weekday()+1],
    'isweekday': [0],
    'isweekend': [0],
    'work_time': [0],

    'y_outdoor': [72],

    'y_zone_temp_2001': [72],
    'y_zone_temp_2002': [72],
    'y_zone_temp_2003': [72],
    'y_zone_temp_2004': [72],
    'y_zone_temp_2005': [72],
    'y_zone_temp_2006': [72],
    
    'hvac_2001': [0], 
    'hvac_2002': [0], 
    'hvac_2003': [0], 
    'hvac_2004': [0], 
    'hvac_2005': [0], 
    'hvac_2006': [0], 
    
    # 'hvac_htg_2001': [72],
    # 'hvac_clg_2001': [72],
    # 'hvac_htg_2002': [72],
    # 'hvac_clg_2002': [72],
    # 'hvac_htg_2003': [72],
    # 'hvac_clg_2003': [72],
    # 'hvac_htg_2004': [72],
    # 'hvac_clg_2004': [72],
    # 'hvac_htg_2005': [72],
    # 'hvac_clg_2005': [72],
    # 'hvac_htg_2006': [72],
    # 'hvac_clg_2006': [72], 
    
    'action_list': [0],
    'reward': [0],
    
    'E_Facility': [0],
    'E_HVAC': [0],
    'E_Heating': [0],
    'E_Cooling': [0],
    
    'T_violation': [0],
    'T_diff': [0],
    'T_mean': [0],
    'T_var': [0],
    
    'sun_is_up': [0],
    'is_raining': [0],
    'outdoor_humidity': [0],
    'wind_speed': [0],
    'diffuse_solar': [0],
        
    'y_humd': [0],
    'y_wind': [0],
    'y_solar': [0]

}



if not os.path.exists('./data/History.csv') or config.reset_dataframe == True:
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(History)
    
    # Save DataFrame to a CSV file
    df.to_csv('./data/History.csv', index=False)
    
    # Save DataFrame to an Excel file
    # df.to_excel('./data/History.xlsx', index=False)
    



import pickle
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # FIFO

    def add(self, state, action, reward, next_state, done):  # add data to buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # sample from buffer with batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(config.buffer_size)
with open(config.replay_buffer, 'wb') as file:
    pickle.dump(replay_buffer, file)







import eppy
from eppy import modeleditor
from eppy.modeleditor import IDF


# # building = idf1.idfobjects['BUILDING'][0]
# # hvac_cool_setpoint = idf1.idfobjects['Schedule:Day:Interval'][0]
# # hvac_heat_setpoint = idf1.idfobjects['Schedule:Day:Interval'][1]
# # hvac_cool_setpoint.Value_Until_Time_1

# # idf1.save()
# Building_Surfaces = idf1.idfobjects['BuildingSurface:Detailed']
# len(Building_Surfaces)

import random

iddfile = config.iddfile
F_top = config.F_top
F_bottom = config.F_bottom

class Building(object):
    
    '''
    Air Wall List:
        Face ['2','12','3','19','11','20','17','28','22','35','29','34','21','42','36','41','10','43']
        Surface Type: Wall
        Construction Name: AirWall
    '''

    def __init__(self, filename_to_run):

        
        IDF.setiddname(iddfile)    
        idf1 = IDF(filename_to_run)
        # print(idf1.idfobjects['BUILDING']) 
        
        self.idf = idf1
        
    def AirWall_Switch(self, switch):
        Building_Surfaces = self.idf.idfobjects['BuildingSurface:Detailed']

        if switch == 'on':
            for i in range(len(Building_Surfaces)): #print(Building_Surfaces[i])
                surface_i = Building_Surfaces[i]
                ''' AirWall '''
                if surface_i['Name'][5::] in ['2','12',
                                              '3','19',
                                              '11','20',
                                              '17','28',
                                              '22','35',
                                              '29','34',
                                              '21','42',
                                              '36','41',
                                              '10','43']:
                    surface_i['Construction_Name'] = 'AirWall'
                    # print(surface_i)
        return (print('AirWall set to True'))
    
    def Roof_Switch(self, switch):
        Building_Surfaces = self.idf.idfobjects['BuildingSurface:Detailed']

        if switch == 'off':
            for i in range(len(Building_Surfaces)): #print(Building_Surfaces[i])
                surface_i = Building_Surfaces[i]
                ''' Roof '''
                if surface_i['Surface_Type'] == 'Roof':
                    surface_i['Sun_Exposure'] = 'NoSun'
                    surface_i['Wind_Exposure'] = 'NoWind'
                    # print(surface_i)
        return (print('Roof set to False'))

    def init_map_2D(self, scale=10):
        self.scale = scale
        self.building_floor = []
        self.floor = []
        self.x_list = []
        self.y_list = []
        self.X = 0
        self.Y = 0
        
        self.zone_center_xy = []
        
        Building_Surfaces = self.idf.idfobjects['BuildingSurface:Detailed']

        for i in range(len(Building_Surfaces)): #print(Building_Surfaces[i])
            surface_i = Building_Surfaces[i]
            
            ''' Floor Map '''
            if surface_i['Surface_Type'] == 'Floor':
                self.building_floor.append([surface_i])

                x1, y1 = surface_i['Vertex_1_Xcoordinate'], surface_i['Vertex_1_Ycoordinate']
                x2, y2 = surface_i['Vertex_2_Xcoordinate'], surface_i['Vertex_2_Ycoordinate']
                x3, y3 = surface_i['Vertex_3_Xcoordinate'], surface_i['Vertex_3_Ycoordinate']
                x4, y4 = surface_i['Vertex_4_Xcoordinate'], surface_i['Vertex_4_Ycoordinate']
                # x5, y5 = surface_i['Vertex_5_Xcoordinate'], surface_i['Vertex_5_Ycoordinate']
                # x6, y6 = surface_i['Vertex_6_Xcoordinate'], surface_i['Vertex_6_Ycoordinate']

                # self.floor.append([[x1,x2,x3,x4], [y1,y2,y3,y4]])
                self.x_list.append([x1,x2,x3,x4])
                self.y_list.append([y1,y2,y3,y4])
                
                
                if surface_i['Vertex_6_Xcoordinate']:
                    x5, y5 = surface_i['Vertex_5_Xcoordinate'], surface_i['Vertex_5_Ycoordinate']
                    x6, y6 = surface_i['Vertex_6_Xcoordinate'], surface_i['Vertex_6_Ycoordinate']
    
                    self.x_list[-1].append(x5)
                    self.x_list[-1].append(x6)

                    self.y_list[-1].append(y5)
                    self.y_list[-1].append(y6)
                    
                    
                    
        for i in range(len(self.x_list)):
            self.x_list[i] = [np.min(self.x_list[i]), np.max(self.x_list[i])]
            self.y_list[i] = [np.min(self.y_list[i]), np.max(self.y_list[i])]

        self.min_X, self.max_X = np.min(self.x_list), np.max(self.x_list) 
        self.min_Y, self.max_Y = np.min(self.y_list), np.max(self.y_list) 
        
        self.x_list = (np.array(self.x_list)-self.min_X) * self.scale
        self.y_list = (np.array(self.y_list)-self.min_Y) * self.scale

        for i in range(len(self.building_floor)):
            self.building_floor[i].append(self.x_list[i])
            self.building_floor[i].append(self.y_list[i])



        self.min_X, self.max_X = np.min(self.x_list), np.max(self.x_list) 
        self.min_Y, self.max_Y = np.min(self.y_list), np.max(self.y_list) 
        
        self.X = round(self.max_X-self.min_X) + 100 
        self.Y = round(self.max_Y-self.min_Y) + 100 
    
        '''Build grid'''
        self.map_2D = np.zeros((self.Y, self.X))
        for i in self.building_floor:
            temp = random.randint(0, 255)
            x1, x2 = np.int32([np.min(i[1]), np.max(i[1])])
            y1, y2 = np.int32([np.min(i[2]), np.max(i[2])])
            
            self.map_2D[y1:y2, x1:x2] = round(temp)
            
        # plt.figure(figsize=(10,10))
        # plt.imshow(self.map_2D)
        return self.map_2D
    

    def draw_map(self, temp):
        self.map_2D[:,:] = temp['Outdoor Temp']
        for i in self.building_floor:
            surface_i = i[0]
            if temp[surface_i['Zone_Name']]:
                zone_temp = temp[surface_i['Zone_Name']]
            else:
                zone_temp = 0

            x1, x2 = np.int32([np.min(i[1]), np.max(i[1])]) +50
            y1, y2 = np.int32([np.min(i[2]), np.max(i[2])]) +50
            
            self.map_2D[y1:y2, x1:x2] = round(zone_temp)
            self.map_2D[y1:y2, x1] = F_bottom
            self.map_2D[y1:y2, x2] = F_bottom
            self.map_2D[y1, x1:x2] = F_bottom
            self.map_2D[y2, x1:x2] = F_bottom
            
            self.zone_center_xy.append([(x1+x2)/2, self.Y-(y1+y2)/2])
            
        # plt.figure(figsize=(10,10))
        # plt.imshow(self.map_2D)
        self.map_2D[0,0] = F_top
        self.map_2D[0,1] = F_bottom
        self.map_2D[:,:] = self.map_2D[::-1,:]

        return self.map_2D, self.zone_center_xy

        


# temp = {}

# temp['Thermal Zone 1'] = 10
# temp['Thermal Zone 2'] = 15
# temp['Thermal Zone 3'] = 100
# temp['Thermal Zone 4'] = 25
# temp['Thermal Zone 5'] = 30
# temp['Thermal Zone 6'] = 35


# map_2D = ITRC_2.draw_map(temp)

###############################################################################
import openstudio
'''
For reproducibility, here are the versions I used to create and run this notebook
_s = !pip list
print(f"Pip package used initially: {[x for x in _s if 'openstudio' in x][0]}")
print(f"OpenStudio Long Version:    {openstudio.openStudioLongVersion()}")    
'''


'''
Openstudio setup
define date, save model

'''
# test_name = "Room Air Zone Vertical Temperature Gradient"
# osm_name_box = 'ITRC_2nd_6zone_OPEN.osm'

current_dir = os.getcwd()
osm_path = os.path.join(current_dir,config.osm_name_box)
osm_path = openstudio.path(osm_path) # I guess this is how it wants the path for the translator
print(osm_path)


translator = openstudio.osversion.VersionTranslator()
osm = translator.loadModel(osm_path).get()

# Create an example model
# m = openstudio.model.exampleModel()
m = translator.loadModel(osm_path).get()

zones = [zone for zone in openstudio.model.getThermalZones(m)]



# Set output variables
[x.remove() for x in m.getOutputVariables()]

o = openstudio.model.OutputVariable("Site Outdoor Air Drybulb Temperature", m)
o.setKeyValue("Environment")
o.setReportingFrequency("Timestep")


for var in ["Site Outdoor Air Drybulb Temperature",
            "Site Wind Speed",
            "Site Wind Direction",
            "Site Solar Azimuth Angle",
            "Site Solar Altitude Angle",
            "Site Solar Hour Angle"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Environment')
    o.setReportingFrequency("Timestep")

# o = openstudio.model.OutputVariable("Zone Thermal Comfort Fanger Model PPD", m)
# o.setKeyValue('*')
# o.setReportingFrequency("Timestep")



for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 1')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 2')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 3')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 4')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 5')
    o.setReportingFrequency("Timestep")
    
for var in ["Zone Air Relative Humidity",
            "Zone Windows Total Heat Gain Energy",
            "Zone Infiltration Mass",
            "Zone Mechanical Ventilation Mass",
            "Zone Mechanical Ventilation Mass Flow Rate",
            "Zone Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Thermostat Heating Setpoint Temperature",
            "Zone Thermostat Cooling Setpoint Temperature"]:
    o = openstudio.model.OutputVariable(var, m)
    o.setKeyValue('Thermal Zone 6')
    o.setReportingFrequency("Timestep")


# Set timestep
timestep = m.getTimestep()
timestep.setNumberOfTimestepsPerHour(config.timestep_per_hour)

# Check the heating thermostat schedule
# z = m.getThermalZones()[2]
# print(z)
# t = z.thermostatSetpointDualSetpoint().get()
# heating_sch = t.heatingSetpointTemperatureSchedule().get()
# o = heating_sch.to_ScheduleRuleset()


# Restrict to one month of simulation
r = m.getRunPeriod()
# print(r)

r.setBeginMonth(config.begin_month)
r.setBeginDayOfMonth(config.begin_day_of_month)

r.setEndMonth(config.end_month)
r.setEndDayOfMonth(config.end_day_of_month)


ft = openstudio.energyplus.ForwardTranslator()
w = ft.translateModel(m)
w.save(openstudio.path(config.save_idf), True)

    

filename_to_run = config.save_idf
IDF.setiddname(iddfile)
IDF.getiddname()
idf1 = IDF(filename_to_run)
idf1.printidf()
# print(idf1.idfobjects['BUILDING']) # put the name of the object you'd like to look at in brackets



'''
Building Model
'''

filename_to_run = config.save_idf

ITRC_2 = Building(config.save_idf)
ITRC_2.idf.idfobjects['BuildingSurface:Detailed']
ITRC_2.AirWall_Switch(config.AirWall_Switch)
ITRC_2.Roof_Switch(config.Roof_Switch)
ITRC_2.idf.saveas(config.save_idf)

THERMAL_MAP_2D = ITRC_2.init_map_2D()
building_floor = ITRC_2.building_floor
# plt.imshow(THERMAL_MAP_2D)

print('save to:', config.save_idf)














