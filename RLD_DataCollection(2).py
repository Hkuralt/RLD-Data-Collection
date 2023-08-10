#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:26:15 2023

@author: haleykuralt
"""
#%%
# Import packages
import numpy as np
import math
import serial
from datetime import datetime 
from matplotlib import pyplot as plt
import ray
import pykinematics as pk
import time


#Filtering 
from scipy.signal import butter,filtfilt
from scipy.integrate import cumtrapz

#%% Initailize variables
# Reqiurements for filter, low pass butterworth filter, 60hz cutoff
fs = 50 #sampling frequency in Hz
recording_duration = 60
samples = fs*recording_duration
BAUDRATE = 9600
cutoff = 0.1
order = 4    
#%%
#Start Ray Instance
ray.shutdown()
ray.init()

# def Timestamp():
#     date_now = time.strftime('%d/%m/%y')
#     time_now = time_now = time.strftime('%H:%M:%S')
#     return [date_now,time_now]

#%% Record Data
@ray.remote
def sensor_data1():
    
    BAUDRATE = 9600 # bit rate [Match arduino code]
    out_folder = '.' # Save data to this folder '.' = current folder
    com_port = '/dev/cu.usbmodem141401' #[Match arduino code]
    sensor_number = 1
    channels = 9
    
    #Set Up Arrays
    print('Initializing arrays and plots...')  
    #Calculate number of samples
    samples = int(fs*recording_duration)
    #Create a matrix [samples x channels]
    sample_data = np.zeros((samples,channels)) 
    #Find size of matrix
    samples,channels = sample_data.shape
    #Record Arduino Data
    def record_arduino():
            try:
                arduino_string = dev.readline().decode('ascii') # this .decode piece converts from byte string to ascii string... probably not necessary.
            except: # sometimes failed for unknown reasons
                
                arduino_string = ''
            #Split incoming data into a list to read individual columns
            # print(arduino_string)
            data_array = arduino_string.split() 
            #Collect al channels of raw data
            data_chan = np.array(data_array[0:channels+1],dtype=float)
            # now = datetime.now()
            # time = now.strftime("%Y-%m-%d_%H-%M-%S")
            # time1 = np.array(time)
            # print(time1)
            return data_chan
    print(f'Opening serial port {com_port} at {BAUDRATE} bps for sensor {sensor_number}')
    print(f'Recording Data for {recording_duration} Seconds')
    with serial.Serial(port=com_port,baudrate=BAUDRATE) as dev:
        # Set up
        dev.flushInput() #Removeold data 
        record_arduino() #Record data 
        #Fill premade array with data
        for sample_index in range(0,samples):
            # keep reading until it's time for the next sample
            sample_data[sample_index,:] = record_arduino()    
    # Saving Data Code
    print('Saving Data')        
    #Out folder = current folder
    out_folder = '.'
    #Get current time for file name
    now = datetime.now()
    datestr = now.strftime("%Y-%m-%d_%H-%M-%S")
    #Save data
    out_data = '%s/Sensor_Data_1_%s.npy'%(out_folder,datestr)
    np.save(out_data,sample_data)
    print('Done!')
    # return sample_data

@ray.remote
def sensor_data2():
    
    BAUDRATE = 9600 # bit rate [Match arduino code]
    out_folder = '.' # Save data to this folder '.' = current folder
    com_port = '/dev/cu.usbmodem141301' #[Match arduino code]
    sensor_number = 2
    channels = 10
    
    # Set Up Arrays
    print('Initializing arrays and plots...')  
    #Calculate number of samples
    samples = int(fs*recording_duration)
    #Create a matrix [samples x channels]
    sample_data = np.zeros((samples,channels)) 
    #Find size of matrix
    samples,channels = sample_data.shape
    # Record Arduino Data
    def record_arduino():
            try:
                arduino_string = dev.readline().decode('ascii') # this .decode piece converts from byte string to ascii string... probably not necessary.
            except: # sometimes failed for unknown reasons
                
                arduino_string = ''
            #Split incoming data into a list to read individual columns
            data_array = arduino_string.split() 
            print(data_array[9])
            #Collect al channels of raw data
            data_chan = np.array(data_array[0:channels+1],dtype=float)
            return data_chan
    print(f'Opening serial port {com_port} at {BAUDRATE} bps for sensor {sensor_number}')
    print(f'Recording Data for {recording_duration} Seconds')
    with serial.Serial(port=com_port,baudrate=BAUDRATE) as dev:
        # Set up
        dev.flushInput() #Removeold data 
        record_arduino() #Record data 
        #Fill premade array with data
        for sample_index in range(0,samples):
            # keep reading until it's time for the next sample
            sample_data[sample_index,:] = record_arduino()    
    # Saving Data Code
    print('Saving Data')        
    #Out folder = current folder
    out_folder = '.'
    #Get current time for file name
    now = datetime.now()
    datestr = now.strftime("%Y-%m-%d_%H-%M-%S")
    #Save data
    out_data = '%s/Sensor_Data_2_%s.npy'%(out_folder,datestr)
    np.save(out_data,sample_data)
    print('Done!')

ray.get([sensor_data1.remote(),sensor_data2.remote()])

#%% Filter Data

def lowpass_filter(s1_data, cutoff, fs, order):
    cutoff = 0.1
    b, a = butter(order, cutoff, btype='low',analog=False)
    sd = filtfilt(b, a, s1_data)
    return sd

# #Right Leg
# s1_data = np.load('/Users/haleykuralt/Library/Mobile Documents/com~apple~CloudDocs/Data/Sensor_Data_1_2023-06-27_11-40-27.npy')
# s2_data = np.load('/Users/haleykuralt/Library/Mobile Documents/com~apple~CloudDocs/Data/Sensor_Data_2_2023-06-27_11-40-29.npy')

#Left Leg
s1_data = np.load('/Users/haleykuralt/Library/Mobile Documents/com~apple~CloudDocs/Data/Sensor_Data_1_2023-06-27_12-01-23.npy')
s2_data = np.load('/Users/haleykuralt/Library/Mobile Documents/com~apple~CloudDocs/Data/Sensor_Data_2_2023-06-27_12-01-26.npy')

s1_a = np.array(s1_data)
s2_all = np.array(s2_data)
s2_a = s2_all[:,0:9]
T_a = s2_all[:,9]

# T_data = np.load('/Users/haleykuralt/Torque_data_2023-06-14_17-43-24.npy')
# T_a = np.array(T_data)
Torq_data = np.array([])
for i in range(len(T_a)):
    
    if (T_a[i]==''):
        T_a[i] = T_a[i-1]  
    Torq_data = np.append(Torq_data, T_a[i]).astype(float)
    if (Torq_data[i].is_integer()):
        Torq_data[i] = Torq_data[i-1]

Torque_data_filtered = lowpass_filter(Torq_data,cutoff,fs,order)

# Unfiltered
s1_auf = s1_a[:,0:3];
s2_auf = s2_a[:,0:3];
s1_guf = s1_a[:,3:6];
s2_guf = s2_a[:,3:6];
s1_muf = s1_a[:,6:9];
s2_muf = s2_a[:,6:10];


#Decompose each matrice
s1_aax = np.array([0][0], dtype = float)
s2_aax = np.array([0][0], dtype = float)
s1_gax = np.array([0][0], dtype = float)
s2_gax = np.array([0][0], dtype = float)
s1_max = np.array([0][0], dtype = float)
s2_max = np.array([0][0], dtype = float)

s1_aay = np.array([0][0], dtype = float)
s2_aay = np.array([0][0], dtype = float)
s1_gay = np.array([0][0], dtype = float)
s2_gay = np.array([0][0], dtype = float)
s1_may = np.array([0][0], dtype = float)
s2_may = np.array([0][0], dtype = float)

s1_aaz = np.array([0][0], dtype = float)
s2_aaz = np.array([0][0], dtype = float)
s1_gaz = np.array([0][0], dtype = float)
s2_gaz = np.array([0][0], dtype = float)
s1_maz = np.array([0][0], dtype = float)
s2_maz = np.array([0][0], dtype = float)


for i in range(len(s1_auf)):
    s1_aax = np.append(s1_aax,s1_auf[i][0])
    s2_aax = np.append(s2_aax,s2_auf[i][0])
    s1_gax = np.append(s1_gax,s1_guf[i][0])
    s2_gax = np.append(s2_gax,s2_guf[i][0])
    s1_max = np.append(s1_max,s1_muf[i][0])
    s2_max = np.append(s2_max,s2_muf[i][0])
    
    
    s1_aay = np.append(s1_aay,s1_auf[i][1])
    s2_aay = np.append(s2_aay,s2_auf[i][1])
    s1_gay = np.append(s1_gay,s1_guf[i][1])
    s2_gay = np.append(s2_gay,s2_guf[i][1])
    s1_may = np.append(s1_may,s1_muf[i][1])
    s2_may = np.append(s2_may,s2_muf[i][1])
    
    s1_aaz = np.append(s1_aaz,s1_auf[i][2])
    s2_aaz = np.append(s2_aaz,s2_auf[i][2])
    s1_gaz = np.append(s1_gaz,s1_guf[i][2])
    s2_gaz = np.append(s2_gaz,s2_guf[i][2])
    s1_maz = np.append(s1_maz,s1_muf[i][2])
    s2_maz = np.append(s2_maz,s2_muf[i][2])


#lowpass filter for each data
s1_accx = lowpass_filter(s1_aax,cutoff,fs,order)
s2_accx = lowpass_filter(s2_aax,cutoff,fs,order)
s1_gyrox = lowpass_filter(s1_gax,cutoff,fs,order)
s2_gyrox = lowpass_filter(s2_gax,cutoff,fs,order)
s1_magx = lowpass_filter(s1_max,cutoff,fs,order)
s2_magx = lowpass_filter(s2_max,cutoff,fs,order)

s1_accy = lowpass_filter(s1_aay,cutoff,fs,order)
s2_accy = lowpass_filter(s2_aay,cutoff,fs,order)
s1_gyroy = lowpass_filter(s1_gay,cutoff,fs,order)
s2_gyroy = lowpass_filter(s2_gay,cutoff,fs,order)
s1_magy = lowpass_filter(s1_may,cutoff,fs,order)
s2_magy = lowpass_filter(s2_may,cutoff,fs,order)


s1_accz = lowpass_filter(s1_aaz,cutoff,fs,order)
s2_accz = lowpass_filter(s2_aaz,cutoff,fs,order)
s1_gyroz = lowpass_filter(s1_gaz,cutoff,fs,order)
s2_gyroz = lowpass_filter(s2_gaz,cutoff,fs,order)
s1_magz = lowpass_filter(s1_maz,cutoff,fs,order)
s2_magz = lowpass_filter(s2_maz,cutoff,fs,order)


sampling_period = 1/fs;

#Recompose each matrix
s1_acc = np.empty([samples,3],dtype=float)
s2_acc = np.empty([samples,3],dtype=float)
s1_gyro = np.empty([samples,3],dtype=float)
s2_gyro = np.empty([samples,3],dtype=float)
s1_mag = np.empty([samples,3],dtype=float)
s2_mag = np.empty([samples,3],dtype=float)


for i in range(samples):
    s1_acc[i] = np.array([s1_accx[i],s1_accy[i],s1_accz[i]], dtype = float)
    s2_acc[i] = np.array([s2_accx[i],s2_accy[i],s2_accz[i]], dtype = float)
    s1_gyro[i] = np.array([s1_gyrox[i],s1_gyroy[i],s1_gyroz[i]], dtype = float)
    s2_gyro[i] = np.array([s2_gyrox[i],s2_gyroy[i],s2_gyroz[i]], dtype = float)
    s1_mag[i] = np.array([s1_magx[i],s1_magy[i],s1_magz[i]], dtype = float)
    s2_mag[i] = np.array([s2_magx[i],s2_magy[i],s2_magz[i]], dtype = float)




#%% SSRO
# initialize the SSRO estimator
ssro = pk.imu.orientation.SSRO(grav = 9.804)
# estimate the state for the whole time series
q_shank_thigh = ssro.run(
    s1_acc,
    s2_acc,
    s1_gyro,
    s2_gyro,
    s1_mag,
    s2_mag,
    fs)  # seconds, so if 100Hz, would be 1/100

# looking at code, it looks like the quaterion output would be the whole state for the Kalman Filter
# because we want only the rotation quaternion (last 4 elements), use q[:, 6:] when passing to the 
# method to extract the rotation matrix
R_shank_thigh = pk.imu.utility.quat2matrix(q_shank_thigh[:, 6:])

#%% Angle Calculation
#Convert rotation matrix to cardan angles
#Conor Suggests using .append to coninue adding elements to an array which is what i need to do for each of the angle calculations
def rot2ang(R_shank_thigh):
    beta = np.array([0], dtype = float);
    gamma = np.array([0], dtype = float);
    alpha = np.array([0], dtype = float);
    for i in range(len(R_shank_thigh)):
        beta = np.append(beta, math.atan2(R_shank_thigh[i,2,1], R_shank_thigh[i,2,2]));
        gamma = np.append(gamma, math.atan2(R_shank_thigh[i,1,0], R_shank_thigh[i,0,0]));
        alpha = np.append(alpha, math.atan2(-R_shank_thigh[i,2,0], np.sqrt((R_shank_thigh[i,0,0])**2 + (R_shank_thigh[i,1,0])**2)));
        i = i + 1
        # print(beta);    
        # print(alpha);
        # print(gamma);
    alpha = alpha[1:len(R_shank_thigh)+1]*(180/np.pi)
    beta = beta[1:len(R_shank_thigh)+1]*(180/np.pi)
    gamma = gamma[1:len(R_shank_thigh)+1]*(180/np.pi)
    #return beta
    ang = np.array([alpha, beta, gamma]);
    return ang
#beta = np.array(beta, dtype = float)
ang = rot2ang(R_shank_thigh);
#beta = rot2ang(R_shank_thigh)


#%% Plot Angles
#Filter cardan angles
angf = lowpass_filter(ang,cutoff,fs,order)

#Initialize variables
x = np.arange(len(R_shank_thigh))
t = np.arange(samples)
time = np.array(t/fs)

#Clean up data
angf[0] = angf[0] - angf[0][5]
angf[1] = angf[1] - angf[1][5]
angf[2] = angf[2] - angf[2][5]

angf[0][:4] = 0
angf[1][:4] = 0
angf[2][:4] = 0

#FOR TRIAL 1 ONLY (I F'ED IT UP)
# angf[0][235:] = 0
# angf[1][235:] = 0
# angf[2][235:] = 0

#Plot cadan angles
plt.figure()
plt.plot(time[0:1500],angf[0,0:1500])
plt.plot(time[0:1500],angf[1,0:1500])
plt.plot(time[0:1500],angf[2,0:1500])
plt.title('Sample Joint Angles')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.legend(['alpha','beta','gamma'])

#Calculate rotation angle
rot_mag = np.sqrt(np.square(angf[1,0:1500]) - np.square(angf[0,0:1500]))
rot = angf[1,0:1500]-angf[0,0:1500]
# rot = rot[:-150]
max_rot = np.max(rot)
min_rot = np.min(rot)

#Plot rotation angle

plt.figure()
plt.plot(time[0:1500], rot)
plt.title('INT-EXT Knee Rotation')
plt.ylabel('Angle (degrees)')
plt.xlabel('Time (s)')

# # may have to play around with plot stuff, but example
# plt.figure()
# plt.plot(T_a, ang[0])


# plt.figure()
# plt.plot(x, ang[1])
#%% Angles based on Gyro only
plt.figure()
plt.plot(x, s2_gyro[:,0])
plt.plot(x, s2_gyro[:,1])
plt.plot(x, s2_gyro[:,2])
plt.legend(['gyro x', 'gyro y' , 'gyro z'])

plt.figure()
plt.plot(x, s1_gyro[:,0])
plt.plot(x, s1_gyro[:,1])
plt.plot(x, s1_gyro[:,2])
plt.legend(['gyro x', 'gyro y' , 'gyro z'])

x_gyro =  s2_gyro[:,0] - s1_gyro[:,0]

plt.figure()
plt.plot(x,x_gyro)

ang_gyro = cumtrapz(x_gyro, dx = 1/fs)

plt.figure()
plt.plot(x[1:], ang_gyro)

#%% Statistical Analysis

#Right Leg
min_rot1 = np.min(rot[:400])
min_rot2 = np.min(rot[400:700])
min_rot3 = np.min(rot[700:])

max_rot1 = np.max(rot[:500])
max_rot2 = np.max(rot[500:850])
max_rot3 = np.max(rot[850:])

#Calculate local minima and maxima
# #Left Leg
# min_rot1 = np.min(rot[:700])
# min_rot2 = np.min(rot[700:1000])
# min_rot3 = np.min(rot[1000:])

# max_rot1 = np.max(rot[:500])
# max_rot2 = np.max(rot[500:850])
# max_rot3 = np.max(rot[850:])

# avg_ext = np.mean([max_rot1, max_rot2, max_rot3])
# avg_int = np.mean([min_rot1, min_rot2, min_rot3])

# SD_ext = np.std([max_rot1, max_rot2, max_rot3])
# SD_int = np.std([min_rot1, min_rot2, min_rot3])
#%% Torque
t = np.arange(samples)
time = np.array(t/fs)
Torque_data_filtered = -Torque_data_filtered
# Torque_data_filtered =  Torque_data_filtered[30:-120]

plt.figure()
plt.plot(time[0:1500],Torque_data_filtered[0:1500])
plt.title('Torque Applied to The Knee')
plt.ylabel('Applied Torque (Nm)')
plt.xlabel('Time (s)')

plt.figure()
plt.plot(rot,Torque_data_filtered[0:1500])
plt.ylabel('Applied Torque (N*m)')
plt.xlabel('Int-Ext Knee Rotation (degrees)')
plt.title('Torque vs. Rotation')


#%%Torque Peaks

#Calculate Local Maxima and Minima
#Right Leg
max_torq1 = np.max(Torque_data_filtered[:500])
max_torq2 = np.max(Torque_data_filtered[500:800])
max_torq3 = np.max(Torque_data_filtered[800:])

min_torq1 = np.min(Torque_data_filtered[:400])
min_torq2 = np.min(Torque_data_filtered[400:700])
min_torq3 = np.min(Torque_data_filtered[700:])

# #Left Leg
# max_torq1 = np.max(Torque_data_filtered[:400])
# max_torq2 = np.max(Torque_data_filtered[400:800])
# max_torq3 = np.max(Torque_data_filtered[800:])

# min_torq1 = np.min(Torque_data_filtered[:700])
# min_torq2 = np.min(Torque_data_filtered[700:1000])
# min_torq3 = np.min(Torque_data_filtered[1000:])

#%% Export to CSV
import pandas as pd

torque_df = pd.DataFrame([time[0:1500],rot,Torque_data_filtered[0:1500]])
torque_df = pd.DataFrame.transpose(torque_df)

# rot_df = pd.DataFrame(rot)

torque_df.to_csv('TorqueData_forMATLAB_Trial2.csv')
# rot_df.to_csv('RotationData_forMATLAB.csv')

#%% Calculate Laxity
poly_curve = np.polyfit(rot, Torque_data_filtered[0:1500], 3)
TR_curve = np.poly1d(poly_curve)
TR_slope = np.poly1d.deriv(TR_curve)
TR_2deriv = np.poly1d.deriv(TR_slope)

plt.figure()
xx = np.linspace(-14.63994474,32.8170845,100)
# plt.plot(rot,Torque_data_filtered[0:1500])
plt.plot(xx,TR_curve(xx), c='r',linestyle='-')
# plt.plot(xx,TR_slope(xx),c='g', linestyle='-')
# plt.plot(xx,TR_2deriv(xx),c='g', linestyle='-')
plt.axvline(9.3,ymin=-10,ymax=10, c='0.5')
plt.axvline(-2.669972368,ymin=-10,ymax=10, c='0.5')
plt.axvline(21.05854225,ymin=-10,ymax=10, c='0.5')
plt.ylabel('Applied Torque (N*m)')
plt.xlabel('Int-Ext Knee Rotation (degrees)')
plt.title()






















