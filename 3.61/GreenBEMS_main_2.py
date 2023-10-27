#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:07:10 2023

@author: MaxGr
"""


import os

import time
import copy
import math
import shutil
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transformtransforms

# from tqdm import tqdm
# from torchvision import models
# from torchsummary import summary
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToPILImage

from collections import namedtuple    
import pytoml
from sqlalchemy import create_engine, text



###############################################################################

def temp_c_to_f(temp_c):#: float, arbitrary_arg=None):
    """Convert temp from C to F. Test function with arbitrary argument, for example."""
    return 1.8 * temp_c + 32

def temp_f_to_c(temp_f):
    return (temp_f-32)/1.8



'''
5 Layer DNN
'''
class DNN_5(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN_5, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim, 128))
        # self.layer2 = nn.Sequential(nn.Linear(128, 512))
        # self.layer3 = nn.Sequential(nn.Linear(256, 512))
        # self.layer4 = nn.Sequential(nn.Linear(512, 256))
        self.layer5 = nn.Sequential(nn.Linear(128, 128))
        self.layer6 = nn.Sequential(nn.Linear(128, action_dim))
        
                
    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        # x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
                
        return x
    
    
    
# device = torch.device("cuda")
# model = DNN_5(5,2).to(device)
# summary(model, (4320,5))


###############################################################################



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



class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        # Q Network
        self.q_net = DNN_5(state_dim, action_dim).to(device)  
        # Target Network
        self.target_q_net = DNN_5(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # discout
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # target update period
        self.count = 0  # record updates
        self.device = device

    def take_action(self, state):  # epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions)  # Q value
        
        # maxQ
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD error
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # MSE
        self.optimizer.zero_grad()  # clear grad
        dqn_loss.backward()  # back-propagation & update network
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict( 
                self.q_net.state_dict())  # update target network
        self.count += 1
        return dqn_loss

    
def HVAC_action(action, temp):
    
        if action == 0:
            H_new = F_bottom
            C_new = F_top
            
        elif action == 1:
            H_new = temp[0]
            C_new = temp[1]
            
        return int(H_new), int(C_new)
    

    

###############################################################################
class Data_Bank():
    
    def __init__(self):
        self.view_distance  = 2000
        self.NUM_HVAC       = 6
        self.FPS            = FPS
        
        self.E_factor_day       = E_factor_day
        self.T_factor_day       = T_factor_day
        
        self.E_factor_night     = E_factor_night
        self.T_factor_night     = T_factor_night
                
        self.episode_reward = 0
        self.episode_return = 0

        self.RL_flag        = RL_flag
        self.time_interval  = 0
        self.time_line      = []
        self.T_Violation    = []
        self.score          = []
        
        self.T_diff         = []
        self.T_maen         = []
        self.T_var          = []
        
        self.T_map          = {}

        
        ''' handles '''
      
        

        ''' time '''
        self.x = []
        
        self.years = []
        self.months = []
        self.days = []
        self.hours = []
        self.minutes = []
        self.current_times = []
        self.actual_date_times = []
        self.actual_times = []
        
        self.weekday = []
        self.isweekday = []
        self.isweekend = []
        self.work_time = []
        
        ''' building parameters '''
        # self.oa_temp_handle        = -1
        self.y_humd = []
        self.y_wind = []
        self.y_solar = []
        
        self.y_zone_humd = []
        self.y_zone_window = []
        self.y_zone_ventmass = []
        
        self.y_zone_temp = []
        
        self.y_outdoor = []
        self.y_zone = []
        self.y_htg = []
        self.y_clg = []
        
        self.y_zone_temp_2001 = []
        self.y_zone_temp_2002 = []
        self.y_zone_temp_2003 = []
        self.y_zone_temp_2004 = []
        self.y_zone_temp_2005 = []
        self.y_zone_temp_2006 = []
        # self.y_zone_temp_2007 = []
        
        self.sun_is_up = []
        self.is_raining = []
        self.outdoor_humidity = []
        self.wind_speed = []
        self.diffuse_solar = []
        
        self.E_Facility = []
        self.E_HVAC = []
        self.E_Heating = []
        self.E_Cooling = []
        
        self.E_HVAC_all = []
        
        ''' DQN '''
        self.action_list = []
        self.episode_reward = []
        
        self.hvac_htg_2001 = []
        self.hvac_clg_2001 = []
        self.hvac_htg_2002 = []
        self.hvac_clg_2002 = []
        self.hvac_htg_2003 = []
        self.hvac_clg_2003 = []
        self.hvac_htg_2004 = []
        self.hvac_clg_2004 = []
        self.hvac_htg_2005 = []
        self.hvac_clg_2005 = []
        self.hvac_htg_2006 = []
        self.hvac_clg_2006 = []
        # self.hvac_htg_2007 = []
        # self.hvac_clg_2007 = []
        return
    



            
def read_parameters_from_txt(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                parameters[key.strip()] = value.strip()
                print(f"{key.strip()}: {value.strip()}")
    return parameters



def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting folder '{folder_path}': {e}")


import csv
import pandas as pd

def output_to_csv(file_path, data):                
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter=',')
    
    # Append the new variables and values to the bottom line
    new_row = pd.DataFrame({
        'date': data['date'],
        'time': data['time'],
        'outdoor temperature': data['outdoor temperature'],
        'indoor temperature 1': data['indoor temperature 1'],
        'indoor temperature 2': data['indoor temperature 2'],
        'indoor temperature 3': data['indoor temperature 3'],
        'indoor temperature 4': data['indoor temperature 4'],
        'indoor temperature 5': data['indoor temperature 5'],
        'indoor temperature 6': data['indoor temperature 6'],
        'setpoint 1': data['setpoint 1'],
        'setpoint 2': data['setpoint 2'],
        'setpoint 3': data['setpoint 3'],
        'setpoint 4': data['setpoint 4'],
        'setpoint 5': data['setpoint 5'],
        'setpoint 6': data['setpoint 6']
    })
    # df = df.append(new_row, ignore_index=True)
    df = pd.concat([df, new_row])
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, sep=',', index=False)

        
###############################################################################
'''
main function
'''


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    TORCH_CUDA_ARCH_LIST="8.6"
    
    # import shutil
    folder_path = "/out"
    delete_folder(folder_path)
    

    '''
    Import parameters from file  
    '''    
    parameters = read_parameters_from_txt('parameters.txt')
    
    EPlus_file = parameters['EPlus_file']
    osm_name_box = parameters['osm_name_box']
    weather_data = parameters['weather_data']
    HVAC_output = parameters['HVAC_output']
    
    timestep_per_hour = int(parameters['timestep_per_hour'])
    begin_month = int(parameters['begin_month'])
    begin_day_of_month = int(parameters['begin_day_of_month'])
    end_month = int(parameters['end_month'])
    end_day_of_month = int(parameters['end_day_of_month'])
    save_idf = parameters['save_idf']
    AirWall_Switch = parameters['AirWall_Switch']
    Roof_Switch = parameters['Roof_Switch']
    RL_flag = bool(parameters['RL_flag'])
    
    state_dim = int(parameters['state_dim'])
    action_dim = int(parameters['action_dim'])

    epochs = int(parameters['epochs'])
    lr = float(parameters['lr'])
    gamma = float(parameters['gamma'])
    epsilon = int(parameters['epsilon'])
    target_update = int(parameters['target_update'])
    buffer_size = int(parameters['buffer_size'])
    minimal_size = int(parameters['minimal_size'])
    batch_size = int(parameters['batch_size'])
    
    FPS = int(parameters['FPS'])
    signal_loss = bool(parameters['signal_loss'])
    signal_factor = float(parameters['signal_factor'])
    T_factor_day = float(parameters['T_factor_day'])
    E_factor_day = float(parameters['E_factor_day'])
    T_factor_night = float(parameters['T_factor_night'])
    E_factor_night = float(parameters['E_factor_night'])
    F_bottom = int(parameters['F_bottom'])
    F_top = int(parameters['F_top'])

    if HVAC_output==True:
        HVAC_file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        
        if not os.path.exists(f'./HVAC_data/{HVAC_file_name}.csv'): 
            columns = ['date', 'time', 'outdoor temperature', 
                      'indoor temperature 1', 'indoor temperature 2','indoor temperature 3', 
                      'indoor temperature 4', 'indoor temperature 5', 'indoor temperature 6',
                      'setpoint 1', 'setpoint 2', 'setpoint 3',
                      'setpoint 4', 'setpoint 5', 'setpoint 6']
            
            df = pd.DataFrame(columns=columns)
    
            # Save the updated DataFrame back to the CSV file
            df.to_csv(f'./HVAC_data/{HVAC_file_name}.csv', sep=',', index=False)
            print("CSV file generated successfully.")


    


    print('torch.version: ',torch. __version__)
    print('torch.version.cuda: ',torch.version.cuda)
    print('torch.cuda.is_available: ',torch.cuda.is_available())
    print('torch.cuda.device_count: ',torch.cuda.device_count())
    print('torch.cuda.current_device: ',torch.cuda.current_device())
    device_default = torch.cuda.current_device()
    torch.cuda.device(device_default)
    print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
    device = torch.device("cuda")

    
    
    HVAC_action_list = []
    for HC_1 in [0,1]:
        for HC_2 in [0,1]:
            for HC_3 in [0,1]:
                for HC_4 in [0,1]:
                    for HC_5 in [0,1]:
                        for HC_6 in [0,1]:
                            HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
        

    
    
    
    
    
    
    
    
    '''
    Hyperparameters for DQN
    
    '''
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # agent.q_net.load_state_dict(torch.load('./weights/Enet_last_19.pth'))
    

    '''
    DQN
    '''
    
    Benchmark = np.zeros((epochs,30), dtype=object)
    
    while True:
        time_start = time.time()
        # print('\n Training iteration: ', epoch)
        print('\n Training step: ', time_start)


    
        EPLUS = Data_Bank()
        # EPLUS.FPS = FPS
        EPLUS.RL_flag = RL_flag

    
        # RL_flag = EPLUS.RL_flag
        # view_distance = EPLUS.view_distance
        # time_interval = EPLUS.time_interval
        # NUM_HVAC = EPLUS.NUM_HVAC
        # FPS = EPLUS.FPS
        # T_factor_day = EPLUS.T_factor_day
        # E_factor_day = EPLUS.E_factor_day
        # T_factor_night = EPLUS.T_factor_night
        # E_factor_night = EPLUS.E_factor_night
        
        
    
        
        '''
        read data from server
        '''
        

        with open("/home/ubuntu/mysql-credentials.toml") as f:
            c = pytoml.load(f)

        engine = create_engine("mysql+pymysql://{user}:{passw}@{host}/{schema}".format(**c))
        print(f"{engine=}")

        Status = namedtuple("Status", "ts building zone vav occupied temperature")
        query = """
SELECT *
  FROM status
  WHERE building = '10.21'
    AND zone = 'L1 NORTH'
    AND (building, zone, ts) IN ( SELECT building, zone, MAX(ts) FROM status )
"""
        results = [ ]
        with engine.connect() as con:
            result = con.execute(text(query))
            # for ts,building,zone,vav,occupied,temperature in result:
            for row in result:
                results.append(Status(*row))

        print(results)

        ''' Time '''
        year = results[0].ts.year
        month = results[0].ts.month
        day = results[0].ts.day
        hour = results[0].ts.hour
        minute = results[0].ts.minute
        current_time = None
        actual_date_time = None
        actual_time = None
        time_step = None
        
        
        '''Temperature'''
        
        # oa_humd      = api.exchange.get_variable_value(state_argument, EPLUS.oa_humd_handle)
        # oa_windspeed = api.exchange.get_variable_value(state_argument, EPLUS.oa_windspeed_handle)
        # oa_winddirct = api.exchange.get_variable_value(state_argument, EPLUS.oa_winddirct_handle)
        # oa_solar_azi = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_azi_handle)
        # oa_solar_alt = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_alt_handle)
        # oa_solar_ang = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_ang_handle)
            
        oa_temp = 75
        
        zone_temp_2001 = results[0].temperature
        zone_temp_2002 = results[1].temperature
        zone_temp_2003 = results[2].temperature
        zone_temp_2004 = results[3].temperature
        zone_temp_2005 = results[4].temperature
        zone_temp_2006 = results[5].temperature
        
        occ = [[60,90], [71,74]]
        hvac_htg_2001 = occ[results[0].occupied][0]
        hvac_clg_2001 = occ[results[0].occupied][1]
        
        hvac_htg_2002 = occ[results[1].occupied][0]
        hvac_clg_2002 = occ[results[1].occupied][1]
        
        hvac_htg_2003 = occ[results[2].occupied][0]
        hvac_clg_2003 = occ[results[2].occupied][1]
        
        hvac_htg_2004 = occ[results[3].occupied][0]
        hvac_clg_2004 = occ[results[3].occupied][1]
        
        hvac_htg_2005 = occ[results[4].occupied][0]
        hvac_clg_2005 = occ[results[4].occupied][1]
        
        hvac_htg_2006 = occ[results[5].occupied][0]
        hvac_clg_2006 = occ[results[5].occupied][1]
        
    
    
        '''
        Store data
        '''
        # EPLUS.y_humd.append(oa_humd)
        # EPLUS.y_wind.append([oa_windspeed,oa_winddirct])
        # EPLUS.y_solar.append([oa_solar_azi, oa_solar_alt, oa_solar_ang])
        # EPLUS.y_zone_humd.append([zone_humd_2001,zone_humd_2002,zone_humd_2003,
        #                           zone_humd_2004,zone_humd_2005,zone_humd_2006
        #                           ])
        
        # EPLUS.y_zone_window.append([zone_window_2001,zone_window_2002,zone_window_2003,
        #                             zone_window_2004,zone_window_2005,zone_window_2006
        #                             ])
        
        # EPLUS.y_zone_ventmass.append([zone_ventmass_2001,zone_ventmass_2002,zone_ventmass_2003,
        #                               zone_ventmass_2004,zone_ventmass_2005,zone_ventmass_2006
        #                               ])
    
    
    
#        EPLUS.y_outdoor.append(temp_c_to_f(oa_temp))
#        
#        EPLUS.y_zone_temp_2001.append(temp_c_to_f(zone_temp_2001))
#        EPLUS.y_zone_temp_2002.append(temp_c_to_f(zone_temp_2002))
#        EPLUS.y_zone_temp_2003.append(temp_c_to_f(zone_temp_2003))
#        EPLUS.y_zone_temp_2004.append(temp_c_to_f(zone_temp_2004))
#        EPLUS.y_zone_temp_2005.append(temp_c_to_f(zone_temp_2005))
#        EPLUS.y_zone_temp_2006.append(temp_c_to_f(zone_temp_2006))
#        
#        EPLUS.hvac_htg_2001.append(temp_c_to_f(hvac_htg_2001))
#        EPLUS.hvac_clg_2001.append(temp_c_to_f(hvac_clg_2001))
#        EPLUS.hvac_htg_2002.append(temp_c_to_f(hvac_htg_2002))
#        EPLUS.hvac_clg_2002.append(temp_c_to_f(hvac_clg_2002))
#        EPLUS.hvac_htg_2003.append(temp_c_to_f(hvac_htg_2003))
#        EPLUS.hvac_clg_2003.append(temp_c_to_f(hvac_clg_2003))
#        EPLUS.hvac_htg_2004.append(temp_c_to_f(hvac_htg_2004))
#        EPLUS.hvac_clg_2004.append(temp_c_to_f(hvac_clg_2004))
#        EPLUS.hvac_htg_2005.append(temp_c_to_f(hvac_htg_2005))
#        EPLUS.hvac_clg_2005.append(temp_c_to_f(hvac_clg_2005))
#        EPLUS.hvac_htg_2006.append(temp_c_to_f(hvac_htg_2006))
#        EPLUS.hvac_clg_2006.append(temp_c_to_f(hvac_clg_2006))
        
    
        T_list = temp_c_to_f(np.array([zone_temp_2001,
                                       zone_temp_2002,
                                       zone_temp_2003,
                                       zone_temp_2004,
                                       zone_temp_2005,
                                       zone_temp_2006]))
        
        EPLUS.y_zone_temp.append(T_list)
        
        
        T_mean = np.mean(T_list)
        
        EPLUS.T_maen.append(T_mean)
        EPLUS.T_diff.append(np.max(T_list)-np.min(T_list))
        EPLUS.T_var.append(np.var(T_list))
            
        # EPLUS.E_Facility.append(api.exchange.get_meter_value(state_argument, EPLUS.E_Facility_handle))
        # EPLUS.E_HVAC.append(api.exchange.get_meter_value(state_argument,     EPLUS.E_HVAC_handle))
        # EPLUS.E_Heating.append(api.exchange.get_meter_value(state_argument,  EPLUS.E_Heating_handle))
        # EPLUS.E_Cooling.append(api.exchange.get_meter_value(state_argument,  EPLUS.E_Cooling_handle))
        
        # EPLUS.E_HVAC_all.append(api.exchange.get_meter_value(state_argument, EPLUS.E_HVAC_handle))
        
        # EPLUS.sun_is_up.append(api.exchange.sun_is_up(state_argument))
        # EPLUS.is_raining.append(api.exchange.today_weather_is_raining_at_time(state_argument,                       hour, time_step))
        # EPLUS.outdoor_humidity.append(api.exchange.today_weather_outdoor_relative_humidity_at_time(state_argument,  hour, time_step))
        # EPLUS.wind_speed.append(api.exchange.today_weather_wind_speed_at_time(state_argument,                       hour, time_step))
        # EPLUS.diffuse_solar.append(api.exchange.today_weather_diffuse_solar_at_time(state_argument,                 hour, time_step))
        
        # api.exchange.today_outdoor_relative_humidity_at_time(state_argument, 19, 2)
        # print(api.exchange.today_weather_wind_direction_at_time(state_argument, hour, time_step))
        
        # Year is bogus, seems to be reading the weather file year instead...         
        # So harcode it to 2022
        # year = 2022
        #EPLUS.years.append(year)
        #EPLUS.months.append(month)
        #EPLUS.days.append(day)
        #EPLUS.hours.append(hour)
        #EPLUS.minutes.append(minute)
        
        #EPLUS.current_times.append(current_time)
        #EPLUS.actual_date_times.append(actual_date_time)
        #EPLUS.actual_times.append(actual_time)
        
        #timedelta = datetime.timedelta()
        #if hour >= 24.0:
            #hour = 23.0
            #timedelta += datetime.timedelta(hours=1)
        #if minute >= 60.0:
            #minute = 59
            #timedelta += datetime.timedelta(minutes=1)
        
        #dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
        #dt += timedelta
        #EPLUS.x.append(dt)
        #EPLUS.time_line.append(dt)
        
        #if dt.weekday() > 4:
            ## print 'Given date is weekend.'
            #EPLUS.weekday.append(dt.weekday())
            #EPLUS.isweekday.append(0)
            #EPLUS.isweekend.append(1)
        #else:
            ## print 'Given data is weekday.'
            #EPLUS.weekday.append(dt.weekday())
            #EPLUS.isweekday.append(1)
            #EPLUS.isweekend.append(0)
            
        #EPLUS.work_time.append(EPLUS.isweekday[-1] * EPLUS.sun_is_up[-1])
        
        
    
        
        
        ''' 
        DQN 
        
        '''
        if EPLUS.RL_flag == False:
            EPLUS.episode_reward.append(0)
    
        if EPLUS.RL_flag == True:
        
            if time_interval == 0:
                EPLUS.episode_reward.append(0)
                EPLUS.action_list.append(0)
        
            '''
            Replay
            '''
            
            done = False
            is_worktime = EPLUS.work_time[-1]
            
            # t_-1
            O0 = EPLUS.y_outdoor[-2]
            E0 = EPLUS.E_HVAC[-2]
            W0 = EPLUS.work_time[-2]
            D0 = EPLUS.weekday[-2]
            M0 = EPLUS.months[-2]
            H0 = EPLUS.hours[-2]
            S0 = EPLUS.sun_is_up[-2]
    
            T_10 = EPLUS.y_zone_temp_2001[-2] 
            T_20 = EPLUS.y_zone_temp_2002[-2] 
            T_30 = EPLUS.y_zone_temp_2003[-2] 
            T_40 = EPLUS.y_zone_temp_2004[-2] 
            T_50 = EPLUS.y_zone_temp_2005[-2] 
            T_60 = EPLUS.y_zone_temp_2006[-2] 
            
            H_10 = EPLUS.hvac_htg_2001[-2] 
            H_20 = EPLUS.hvac_htg_2002[-2] 
            H_30 = EPLUS.hvac_htg_2003[-2] 
            H_40 = EPLUS.hvac_htg_2004[-2] 
            H_50 = EPLUS.hvac_htg_2005[-2] 
            H_60 = EPLUS.hvac_htg_2006[-2] 
            
            state_0 = [O0/100,T_30/100,W0,
                       T_10/100,T_20/100,T_30/100,T_40/100,T_50/100,T_60/100,
                       H_10/100,H_20/100,H_30/100,H_40/100,H_50/100,H_60/100]
            
            # print(state_0)
    
           
            action_0 = EPLUS.action_list[-1]
            
            # t_0
            O1 = EPLUS.y_outdoor[-1] 
            E1 = EPLUS.E_HVAC[-1]
            W1 = EPLUS.work_time[-1]
            D1 = EPLUS.weekday[-1]
            M1 = EPLUS.months[-1]
            H1 = EPLUS.hours[-1]
            S1 = EPLUS.sun_is_up[-1]
    
            T_11 = EPLUS.y_zone_temp_2001[-1] 
            T_21 = EPLUS.y_zone_temp_2002[-1] 
            T_31 = EPLUS.y_zone_temp_2003[-1] 
            T_41 = EPLUS.y_zone_temp_2004[-1] 
            T_51 = EPLUS.y_zone_temp_2005[-1] 
            T_61 = EPLUS.y_zone_temp_2006[-1] 
            
            H_11 = EPLUS.hvac_htg_2001[-1]
            H_21 = EPLUS.hvac_htg_2002[-1] 
            H_31 = EPLUS.hvac_htg_2003[-1] 
            H_41 = EPLUS.hvac_htg_2004[-1] 
            H_51 = EPLUS.hvac_htg_2005[-1]
            H_61 = EPLUS.hvac_htg_2006[-1]
            
            
            # t_1
            state_1 = [O1/100,T_31/100,W1,
                       T_11/100,T_21/100,T_31/100,T_41/100,T_51/100,T_61/100,
                       H_11/100,H_21/100,H_31/100,H_41/100,H_51/100,H_61/100] 
            
            # print(state_1)
    
            action_1 = agent.take_action(state_1)
            action_map = HVAC_action_list[action_1]
    
    
    
            set_temp = [71,74]
    
            # Take action
            H_new_1, C_new_1 = HVAC_action(action_map[0], set_temp)
            H_new_2, C_new_2 = HVAC_action(action_map[1], set_temp)
            H_new_3, C_new_3 = HVAC_action(action_map[2], set_temp)
            H_new_4, C_new_4 = HVAC_action(action_map[3], set_temp)
            H_new_5, C_new_5 = HVAC_action(action_map[4], set_temp)
            H_new_6, C_new_6 = HVAC_action(action_map[5], set_temp)
            

            EPLUS.action_list.append(action_1)
        
    
            
            if HVAC_output==True:
        
                data = {
                    'date': dt.strftime("%Y-%m-%d"),
                    'time': dt.strftime("%H:%M"),
                    'outdoor temperature': temp_c_to_f(oa_temp),
                    'indoor temperature 1': temp_c_to_f(zone_temp_2001),
                    'indoor temperature 2': temp_c_to_f(zone_temp_2002),
                    'indoor temperature 3': temp_c_to_f(zone_temp_2003),
                    'indoor temperature 4': temp_c_to_f(zone_temp_2004),
                    'indoor temperature 5': temp_c_to_f(zone_temp_2005),
                    'indoor temperature 6': temp_c_to_f(zone_temp_2006),
                    'setpoint 1': (H_new_1, C_new_1),
                    'setpoint 2': (H_new_2, C_new_2),
                    'setpoint 3': (H_new_3, C_new_3),
                    'setpoint 4': (H_new_4, C_new_4),
                    'setpoint 5': (H_new_5, C_new_5),
                    'setpoint 6': (H_new_6, C_new_6),
                    }
                
                output_to_csv(f'./HVAC_data/{HVAC_file_name}.csv', data)
    
            
    
            ''' 
            reward define 
            
            '''
            if is_worktime:
                E_factor = E_factor_day
                T_factor = T_factor_day
                work_flag = 0
                reward_signal = 0
                
                # E_save = E_factor
                # T_save = T_factor
                
            else:
                E_factor = E_factor_night
                T_factor = T_factor_night
                work_flag = 0
                reward_signal = 0
    
                    
            # 1 Energy
            reward_E = -E1 * E_factor
            
            # 2 Temp
            if 68<T_11<77:
                reward_T1 = 1*work_flag
            else:
                reward_T1 = -(T_11-72)**2 * T_factor 
                
            if 68<T_21<77:
                reward_T2 = 1*work_flag
            else:
                reward_T2 = -(T_21-72)**2 * T_factor 
                
            if 68<T_31<77:
                reward_T3 = 1*work_flag
            else:
                reward_T3 = -(T_31-72)**2 * T_factor
    
            if 68<T_41<77:
                reward_T4 = 1*work_flag
            else:
                reward_T4 = -(T_41-72)**2 * T_factor
    
            if 68<T_51<77:
                reward_T5 = 1*work_flag
            else:
                reward_T5 = -(T_51-72)**2 * T_factor 
    
            if 68<T_61<77:
                reward_T6 = 1*work_flag
            else:
                reward_T6 = -(T_61-72)**2 * T_factor    
            
            
            # reward_T = np.mean([reward_T1,reward_T2,reward_T3,reward_T4,reward_T5,reward_T6])
            # reward_T = np.sum([reward_T1,reward_T2,reward_T3,reward_T4,reward_T5,reward_T6])
            reward_T = reward_T1+reward_T2+reward_T3+reward_T4+reward_T5+reward_T6
    
                
            # 4 Smootheness
            current_action = HVAC_action_list[EPLUS.action_list[-1]]
            last_action    = HVAC_action_list[EPLUS.action_list[-2]]
            
            change_action = np.array(current_action) ^ np.array(last_action)
            num_unstable = len(change_action[change_action==1])
            reward_signal = -signal_factor * num_unstable
            
            
            if signal_loss == True:
                reward_1 = reward_T + reward_E + reward_signal
            else:
                reward_1 = reward_T + reward_E
        
            EPLUS.episode_reward.append(reward_1)
            EPLUS.episode_return = EPLUS.episode_return + reward_1
            
            
            if is_worktime:
                if T_mean > 77:
                    EPLUS.T_Violation.append(T_mean-77)
                elif T_mean < 68:
                    EPLUS.T_Violation.append(68-T_mean)
                
    
    
    
            """
            if H_new_1<0 or H_new_1>120:
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2001_handle, temp_f_to_c(72))
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2001_handle, temp_f_to_c(72))
                done = True
                print('Temp violation, reseting...')
            if H_new_2<0 or H_new_2>120:
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2002_handle, temp_f_to_c(72))
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2002_handle, temp_f_to_c(72))
                done = True
                print('Temp violation, reseting...')
            if H_new_3<0 or H_new_3>120:
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle, temp_f_to_c(72))
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle, temp_f_to_c(72))
                done = True
                print('Temp violation, reseting...')
            if H_new_4<0 or H_new_4>120:
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2004_handle, temp_f_to_c(72))
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2004_handle, temp_f_to_c(72))
                done = True
                print('Temp violation, reseting...')
            if H_new_5<0 or H_new_5>120:
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2005_handle, temp_f_to_c(72))
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2005_handle, temp_f_to_c(72))
                done = True
                print('Temp violation, reseting...')
            if H_new_6<0 or H_new_6>120:
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2006_handle, temp_f_to_c(72))
                api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2006_handle, temp_f_to_c(72))
                done = True
                print('Temp violation, reseting...')
    
            """
            
    
    
    
            if done == True:
                EPLUS.score.append(EPLUS.episode_return)
                EPLUS.episode_return = 0
                
                
        
            '''
            replay buffer
            
            '''
            # add to experience
            replay_buffer.add(state_0, action_0, reward_1, state_1, done)
            # episode_return += reward_1
            
            
        
            '''
            training
            
            '''
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                                    'states': b_s,
                                    'actions': b_a,
                                    'next_states': b_ns,
                                    'rewards': b_r,
                                    'dones': b_d
                                    }
                agent.update(transition_dict)
    
    

    
    
    
    
    
    
    
    
        torch.save(agent.target_q_net.state_dict(), f'./weights/Enet_last_{epoch}.pth')
    
        E_HVAC_all_DQN = copy.deepcopy(EPLUS.E_HVAC_all)
        
        time_end = time.time()
        time_round = time_end-time_start
        print(f'Training iteration {epoch} finished, total time cost: {time_round}')
            
            
        # plt.figure(figsize=(30,10), dpi=100)
        # plt.plot(E_HVAC_all_RBC, label='Default')
        # plt.plot(E_HVAC_all_DQN, label='DQN')
        # plt.legend()
        
        
        
        work_time_length = EPLUS.work_time.count(1)
        
        
        # work_time_length/len(E_HVAC_all_RBC)
        
        T_violation = len(EPLUS.T_Violation)/ len(EPLUS.x)
        T_violation_offset = np.mean(EPLUS.T_Violation)
        
        print(f'Energy saving ratio: {E_save}')
        print(f'Temperature violation: {T_violation}')
        print(f'Temperature violation offset: {T_violation_offset}')
        
        
        
        Benchmark[epoch, 0] = epoch
        Benchmark[0, 1] = E_HVAC_all_RBC
        Benchmark[epoch, 2] = E_HVAC_all_DQN
        Benchmark[epoch, 3] = E_save
        Benchmark[epoch, 4] = T_violation
        Benchmark[epoch, 5] = T_violation_offset
        Benchmark[epoch, 6] = EPLUS.T_Violation
        Benchmark[epoch, 7] = EPLUS.episode_reward
        Benchmark[epoch, 8] = EPLUS.action_list
        Benchmark[epoch, 9] = time_round
        # Benchmark[epoch, 10] = EPLUS.score
        Benchmark[0, 11] = EPLUS.time_line
        Benchmark[0, 12] = EPLUS.months
        Benchmark[0, 13] = EPLUS.y_outdoor
        # Benchmark[epoch, 14] = EPLUS.y_zone_temp_2003
        # Benchmark[epoch, 15] = np.array(EPLUS.E_HVAC)
        # Benchmark[epoch, 16] = EPLUS.T_maen
        # Benchmark[epoch, 17] = EPLUS.T_diff
        # Benchmark[epoch, 18] = EPLUS.T_var
        # Benchmark[epoch, 19] = np.sum(EPLUS.E_Facility)
        # Benchmark[epoch, 20] = np.array(EPLUS.y_zone_temp)
        # Benchmark[0, 21] = EPLUS.y_humd
        # Benchmark[epoch, 22] = np.array(EPLUS.y_wind)
        # Benchmark[epoch, 23] = np.array(EPLUS.y_solar)
        # Benchmark[epoch, 24] = np.array(EPLUS.y_zone_humd)
        # Benchmark[epoch, 25] = np.array(EPLUS.y_zone_window)
        # Benchmark[epoch, 26] = np.array(EPLUS.y_zone_ventmass)
        Benchmark[0, 27] = EPLUS.work_time
        # Benchmark[epoch, 28] = EPLUS.E_Heating
        # Benchmark[epoch, 29] = EPLUS.E_Cooling
    
        np.save('./Benchmark_data/Benchmark.npy', Benchmark, allow_pickle=True)
        print('Result has been saved...\n')
        
        
        
    
    
    
    
    





