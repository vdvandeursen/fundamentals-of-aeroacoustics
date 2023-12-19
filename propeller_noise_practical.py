# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:21:00 2023

@author: Martijn
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

p_0 = 101325.0
rho_0 = 1.225

c = np.sqrt(1.4 * 287.05 * 288.15)

force_magnitude = 2000
gamma = 0.0

R_0 = 100.0
phi = 0.0

R_1 = 0.85 * 1.0 / 2

blade_numbers = [4]#[4, 5]
mach_numbers = [0.3]#[0.3, 0.6]

thetas = np.radians([0, 15, 30, 45, 60, 75, 90])

n_steps = 1000


for blade_number, mach_number in itertools.product(blade_numbers, mach_numbers):
    Omega = mach_number * c / R_1
    
    for theta in thetas:
        receiver_position = R_0 * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    
        SPL_list = []
        travel_time_list = []
    
        for t in np.linspace(0, 2 * np.pi / Omega, n_steps):
            force_position = np.array([R_1 * np.cos(Omega * t), R_1 * np.sin(Omega * t), 0])
            
            r = receiver_position - force_position
            r_hat = r / np.linalg.norm(r)
    
            F = force_magnitude * np.array([-np.sin(gamma) * np.sin(Omega * t), np.sin(gamma) * np.cos(Omega * t), np.cos(gamma)])
            M = mach_number * np.array([-np.sin(Omega * t), np.cos(Omega * t), 0])
            
            F_dot = force_magnitude * np.array([-Omega * np.sin(gamma) * np.cos(Omega * t), -Omega * np.sin(gamma) * np.sin(Omega * t), 0])
            M_dot = mach_number * np.array([-Omega * np.cos(Omega * t), -Omega * np.sin(Omega * t), 0])
            
            Fr = np.dot(F, r_hat)
            Mr = np.dot(M, r_hat)
            
            F_dot_r = np.dot(F_dot, r_hat)
            M_dot_r = np.dot(M_dot, r_hat)
            
            FM = 0
            
            r = np.linalg.norm(r)
            
            SPL_1 = -1/c * (F_dot_r / (r * (1 - Mr)**2))
            SPL_2 = -1/c * (Fr * (r * M_dot_r + c * (Mr - mach_number)**2) / (r**2 * (1 - Mr)**3))
            SPL_3 = -(Fr - FM) / (r**2 * (1 - Mr)**2)
            
            SPL = (SPL_1 + SPL_2 + SPL_3) / (4.0 * np.pi)
            SPL_list.append(SPL)
            
            travel_time = r / c
            travel_time_list.append(travel_time)

        plt.figure()
        # plt.plot(np.linspace(0, 2 * np.pi / Omega, n_steps), SPL_list) # retarted time vs sound power level
        plt.plot(np.linspace(0, 2 * np.pi / Omega, n_steps) + np.array(travel_time_list), SPL_list) # advanced time vs sound power level
        plt.xlabel("advanced time [s]")
        plt.ylabel("PWL []")
        plt.show()






