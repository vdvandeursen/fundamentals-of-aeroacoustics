import numpy as np
import matplotlib.pyplot as plt
import itertools

# Constants
p0 = 101325  # Pa
rho0 = 1.125  # kg/m3
c = 343  # m/s

force_magnitude = 2000  # N
gamma = 0  # rad
rotor_diameter = 1  # m
rotor_radius = rotor_diameter / 2  # rotor radius
force_radius = 0.85 * rotor_radius  # point of force application
blade_numbers = [4, 5]
mach_numbers = [3, 6]

thetas = np.radians(np.arange(0, 95, 15))  # radians
receiver_distance = 100  # m


n_steps = 50
p_list = []


for blade_number, mach_number in itertools.product(blade_numbers, mach_numbers):
    omega = mach_number * c / force_radius  # Hz

    for theta in thetas:
        receiver_position = receiver_distance * np.array([0, np.sin(theta), np.cos(theta)])  # Cartesian (x,y,z)

        for t in np.linspace(0, 1/omega, n_steps):
            force_position = force_radius * np.array([np.cos(omega*t), np.sin(omega*t), 0])  # Cartesian (x,y,z)

            F = force_magnitude * np.array([0, 0, 1])
            M = mach_number * np.array([-np.sin(omega*t), np.cos(omega*t), 0])
            M_dot = mach_number * np.array([-omega * np.cos(omega*t), -omega * np.sin(omega*t), 0])

            r = receiver_position - force_position
            r_hat = r / np.linalg.norm(r)

            F_dot = np.array([0, 0, 0])  # todo

            M_r = np.dot(M, r_hat)
            F_r = np.dot(F, r_hat)
            M_dot_r = np.dot(M_dot, r_hat)
            F_dot_r = np.dot(F_dot, r_hat)

            F_M = 0

            r = np.linalg.norm(r)

            first = -1/c * (F_dot_r / (r*(1-M_r)**2))
            second = - 1/c * ((F_r*(r*M_dot_r + c*(M_r - mach_number**2)))/(r**2*(1-M_r)**3))
            third = - (F_r - F_M)/(r**2*(1-M_r)**2)

            p = 1/(4*np.pi) * (first + second + third)

            p_list.append(p)

        plt.figure()
        plt.plot(list(np.linspace(0, 1/omega, n_steps)), p_list)
        plt.show()


