import numpy as np
from scipy import special
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import itertools
import PyOctaveBand

# Constants
p0 = 101325  # Pa
rho0 = 1.125  # kg/m3
c = 340.3  # m/s

P_ref = 1E-12  # W
p_ref = 2E-5  # Pa

gamma = 0  # rad
rotor_diameter = 1  # m
rotor_radius = rotor_diameter / 2  # rotor radius
force_radius = 0.85 * rotor_radius  # point of force application


def cosspace(start, stop, n):
    # returns an array with a higher density of points near the ends of the interval

    t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))

    return start + (stop - start) * t


def calculate_pressure_dipole(
        t,
        omega,
        receiver_distance,
        phi,
        force_magnitude,
        harmonic_loading,
        theta,
        tip_mach_number
):
    """Calculate the pressure caused by a single dipole"""
    receiver_position = receiver_distance * np.array(
        [np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
         np.cos(theta)]
    )  # Cartesian (x,y,z)

    force_position = force_radius * np.array([
        np.cos(omega * t),
        np.sin(omega * t),
        0]
    )  # Cartesian (x,y,z)

    mach_number = force_radius / rotor_radius * tip_mach_number

    # Function for the periodic force
    if harmonic_loading:
        periodic_force = force_magnitude * np.sin(omega * t)
        periodic_force_dot = omega * force_magnitude * np.cos(omega*t)
    else:
        periodic_force = force_magnitude
        periodic_force_dot = 0

    # Unit vector r_hat
    r = receiver_position - force_position
    r_hat = r / np.linalg.norm(r)

    F = periodic_force * np.array(
        [-np.sin(gamma) * np.sin(omega * t), np.sin(gamma) * np.cos(omega * t), np.cos(gamma)])
    M = mach_number * np.array([-np.sin(omega * t), np.cos(omega * t), 0])

    # Product rule
    F_dot = periodic_force * np.array(
        [-np.sin(gamma) * omega * np.cos(omega * t), -omega * np.sin(gamma) * np.sin(omega * t), 0]) + \
            periodic_force_dot * np.array(
        [-np.sin(gamma) * np.sin(omega * t), np.sin(gamma) * np.cos(omega * t), np.cos(gamma)])

    M_dot = mach_number * np.array([-omega * np.cos(omega * t), -omega * np.sin(omega * t), 0])

    F_r = np.dot(F, r_hat)
    M_r = np.dot(M, r_hat)

    F_dot_r = np.dot(F_dot, r_hat)
    M_dot_r = np.dot(M_dot, r_hat)

    F_M = 0

    r = np.linalg.norm(r)

    p_1 = -1 / c * (F_dot_r / (r * (1 - M_r) ** 2))
    p_2 = - 1 / c * ((F_r * (r * M_dot_r + c * (M_r - mach_number ** 2))) / (r ** 2 * (1 - M_r) ** 3))
    p_3 = - (F_r - F_M) / (r ** 2 * (1 - M_r) ** 2)

    p = 1 / (4 * np.pi) * (p_1 + p_2 + p_3)
    travel_time = r / c

    return travel_time, p


def calculate_spl_rotor_in_time_domain(
        blade_number,
        tip_mach_number,
        theta,
        thrust,
        receiver_distance,
        plot_pressure,
        harmonic_loading,
        n_steps=100):
    """ Calculate rotor noise (SPL, PWL) using time domain approach"""
    omega = tip_mach_number * c / rotor_radius  # rad/s

    if plot_pressure:
        _, ax = plt.subplots(1)

    calculate_pressure_dipole_vectorized = np.vectorize(
        calculate_pressure_dipole,
        excluded=['omega', 'receiver_distance', 'phi', 'force_magnitude', 'theta', 'tip_mach_number']
    )

    blade_receiver_times = []
    p_blades = []

    # Calculate pressure contribution for each blade in time domain approach
    for B in range(0, blade_number):
        emission_time = np.linspace(0, 6 * np.pi / omega, n_steps)  # simulate each dipole 3 full rotations
        travel_time_blade, p_blade = calculate_pressure_dipole_vectorized(
            emission_time,
            omega=omega,
            phi=B * 2 * np.pi / blade_number,
            receiver_distance=receiver_distance,
            force_magnitude=thrust / blade_number,
            harmonic_loading=harmonic_loading,
            theta=theta,
            tip_mach_number=tip_mach_number
        )
        receiver_time_blade = emission_time + travel_time_blade

        blade_receiver_times.append(receiver_time_blade)
        p_blades.append(p_blade)  # Pressure due to 'blade' aka dipole

    # sample dipole pressure contributions between 1 and 2 rotor rotations
    time_receiver_lower_bound = 2 * np.pi / omega + receiver_distance / c
    time_receiver_upper_bound = 4 * np.pi / omega + receiver_distance / c
    time_receiver = np.linspace(time_receiver_lower_bound, time_receiver_upper_bound, n_steps)

    p_receiver = 0
    for i, blade_receiver_time in enumerate(blade_receiver_times):

        cs = CubicSpline(blade_receiver_time, p_blades[i], bc_type='not-a-knot')
        p = cs(time_receiver)

        p_receiver += p  # Dipoles are coherent so we do this.

        if plot_pressure:
            ax.plot(
                time_receiver,
                p,
                label=f'Blade {i + 1}',
                linestyle='dotted'
            )

    p_receiver_fluctuations = p_receiver - np.mean(p_receiver)

    p_rms = np.sqrt(np.mean(p_receiver_fluctuations ** 2))
    SPL = 10 * np.log10(p_rms ** 2 / p_ref ** 2)
    PWL = SPL + 11 + 20 * np.log10(receiver_distance)

    # Create plots if necessary
    if plot_pressure:

        ax.plot(
            time_receiver,
            p_receiver,
            label='Receiver',
            linewidth=2
        )

        ax.set_title(rf'$\theta$={np.degrees(theta):.2f}$^\circ$')
        ax.legend()
        ax.set_xlabel("Receiver time [s]")
        ax.set_ylabel("Pressure [Pa]")

        ax.set_ylim(-0.1, 0.1)

        plt.show()

    return SPL, PWL, time_receiver, p_receiver_fluctuations


def calculate_spl_rotor_in_frequency_domain(
        theta,
        receiver_distance,
        blade_number,
        tip_mach_number,
        thrust,
        harmonic_loading,
        phi,
        return_spl_only=False
):
    """ Calculate rotor noise (SPL, PWL) using frequency domain approach"""
    omega = tip_mach_number * c / rotor_radius  # rad/s
    B = blade_number
    R0 = receiver_distance
    mach_number = force_radius / rotor_radius * tip_mach_number

    SPL_mb = []  # SPL of each B*omega harmonic
    freqs = []
    p_mb_s = []

    for k in range(1, 20):
        p_mb = 0

        for m in [-k, k]:
            if harmonic_loading:
                # We've defined the harmonic loading as F(t) = thrust*sin(omega t)
                s_values = [-1, 1]
                Fs_values = [thrust/blade_number * -1j, thrust/blade_number * 1j]

            else:
                s_values = [0]
                Fs_values = [thrust / blade_number]

            fraction = (-1j * m * B ** 2 * omega * np.exp(-1j * m * B * omega * R0 / c)) / (4 * np.pi * c * R0)
            summation = 0

            for s, Fs in zip(s_values, Fs_values):
                summation += Fs * np.exp(-1j * (m * B - s) * (phi - np.pi / 2)) * special.jv(m * B - s,
                                                                                             m * B * mach_number * np.sin(
                                                                                                 theta)) * (
                                     -(m * B - s) / (m * B) * (np.sin(gamma) / mach_number) + np.cos(theta) * np.cos(
                                 gamma)
                             )

            p_mb += fraction * summation

            # p_mb += -1j * m * B ** 2 * omega / (4*np.pi*c*R0) * Fs * np.exp(1j * m * B * np.pi/2) * \
            #       np.exp(-1j * m * B * (phi + omega * R0 / c)) * special.jv(m*B, m*B*mach_number * np.sin(theta)) * \
            #       np.cos(theta)

        p_mb_rms = p_mb / np.sqrt(2)  # Root means squared pressure for each harmonic
        SPL_mb.append(10 * np.log10(p_mb_rms ** 2 / p_ref ** 2))  # SPL for each harmonic
        freqs.append(m*blade_number*omega/2/np.pi)

    SPL = 10 * np.log10(sum([10**(SPL/10) for SPL in SPL_mb]))  # Harmonics are incoherent, so summation goes like this
    PWL = SPL + 11 + 20*np.log10(receiver_distance)

    PWL_s = np.array(SPL_mb) + 11 + 20*np.log10(receiver_distance)

    if return_spl_only:
        return SPL

    return SPL, PWL, freqs, PWL_s
