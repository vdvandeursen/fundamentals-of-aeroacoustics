import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import itertools

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
blade_numbers = [4, 5]
mach_numbers = [.3, .6]

thetas = np.radians(range(0, 91, 15))  # radians
receiver_distance = 100  # m


def calculate_pressure_dipole(t, omega, receiver_distance, phi, force_magnitude):
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

    # Function for the periodic force
    periodic_force = force_magnitude + 0.5 * force_magnitude * np.sin(omega * t)
    periodic_force = force_magnitude

    # Unit vector r_hat
    r = receiver_position - force_position
    r_hat = r / np.linalg.norm(r)

    F = periodic_force * np.array(
        [-np.sin(gamma) * np.sin(omega * t), np.sin(gamma) * np.cos(omega * t), np.cos(gamma)])
    M = mach_number * np.array([-np.sin(omega * t), np.cos(omega * t), 0])

    F_dot = periodic_force * np.array(
        [-np.sin(gamma) * omega * np.cos(omega * t), -omega * np.sin(gamma) * np.sin(omega * t), 0])

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


def calculate_spl_rotor_in_time_domain(blade_number, mach_number, theta, thrust, plot_pressure):
    omega = mach_number * c / force_radius  # rad/s
    n_steps = 1000

    if plot_pressure:
        _, ax = plt.subplots(1)

    time_domain_approach = np.vectorize(calculate_pressure_dipole,
                                        excluded=['omega', 'receiver_distance', 'phi', 'force_magnitude'])

    blade_receiver_times = []
    p_blades = []

    SPLs = []

    # Calculate pressure contribution for each blade in time domain approach
    for B in range(0, blade_number):
        emission_time = np.linspace(0, 6 * np.pi / omega, n_steps)  # simulate each dipole 3 full rotations
        travel_time_blade, p_blade = time_domain_approach(
            emission_time,
            omega=omega,
            phi=B * 2 * np.pi / blade_number,
            receiver_distance=receiver_distance,
            force_magnitude=thrust / blade_number
        )
        receiver_time_blade = emission_time + travel_time_blade

        blade_receiver_times.append(receiver_time_blade)
        p_blades.append(p_blade)

        # p_rms_blade = np.sqrt(np.mean((p_blade - np.mean(p_blade)) ** 2))
        # SPL_blade = 10 * np.log10(p_rms_blade ** 2 / p_ref ** 2)
        #
        # SPLs.append(SPL_blade)

    # sample dipole pressure contributions between 1 and 2 rotor rotations
    time_receiver_lower_bound = 2 * np.pi / omega + receiver_distance / c
    time_receiver_upper_bound = 4 * np.pi / omega + receiver_distance / c
    time_receiver = np.linspace(time_receiver_lower_bound, time_receiver_upper_bound, n_steps)

    p_receiver = 0
    for i, (receiver_time_blade, p_blade) in enumerate(zip(blade_receiver_times, p_blades)):
        p = np.interp(
            x=time_receiver,
            xp=receiver_time_blade,
            fp=p_blade
        )

        p_rms_blade = np.sqrt(np.mean((p - np.mean(p)) ** 2))
        SPL_blade = 10 * np.log10(p_rms_blade ** 2 / p_ref ** 2)
        SPLs.append(SPL_blade)

        p_receiver += p

        if plot_pressure:
            ax.plot(
                time_receiver,
                p,
                label=f'Blade {i + 1}',
                linestyle='dotted'
            )

    # SPL = 10*np.log10(sum([10**(SPL/10) for SPL in SPLs]))
    # PWL = SPL + 11 + 20*np.log10(receiver_distance)

    p_rms = np.sqrt(np.mean((p_receiver - np.mean(p_receiver)) ** 2))
    SPL = 10 * np.log10(p_rms ** 2 / p_ref ** 2)
    PWL = SPL + 11 + 20 * np.log10(receiver_distance)

    # I = p_rms ** 2 / (rho0 * c)
    # P = I * receiver_distance ** 2 * 4 * np.pi
    #
    # PWL = 10 * np.log10(P / P_ref)

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

    return SPL, PWL


def calculate_spl_rotor_in_frequency_domain(blade_number, mach_number, theta, thrust):
    omega = mach_number * c / force_radius  # rad/s
    B = blade_number
    R0 = receiver_distance
    Fs = thrust / blade_number
    # Fs = thrust

    P = []
    freq = []

    phi = 0

    for k in range(1, 5):
        p_mb = 0

        for m in [-k, k]:
            p_mb += -1j * m * B ** 2 * omega / (4*np.pi*c*R0) * Fs * np.exp(1j * m * B * np.pi/2) * \
                      np.exp(-1j * m * B * (phi + omega * R0 / c)) * special.jv(m*B, m*B*mach_number * np.sin(theta)) * \
                      np.cos(theta)

        freq.append(m*B*omega/2/np.pi)  # Hz
        P.append(p_mb)  # unit is Pa / Hz?

    P = np.array(P)
    freq = np.array(freq)

    power = 1/(2*np.pi) * np.sum(np.abs(P) * freq)  # parsevals theorem

    PWL = 10 * np.log10(power / P_ref)

    # PWL_1 = 10*np.log10(power / p_ref)
    # SPL = 10*np.log10(np.sum(np.abs(P)/p_ref**2))

    SPL = PWL - 11 - 20*np.log10(receiver_distance)

    return SPL, PWL


if __name__ == "__main__":
    for blade_number, mach_number in itertools.product(blade_numbers, mach_numbers):
        SPL_t_list = []
        PWL_t_list = []
        SPL_f_list = []
        PWL_f_list = []

        thetas = np.linspace(0, np.pi/2, 30)

        for plot_num, theta in enumerate(thetas):
            SPL_t, PWL_t = calculate_spl_rotor_in_time_domain(
                blade_number=blade_number,
                mach_number=mach_number,
                theta=theta,
                thrust=2000,
                plot_pressure=False
            )

            SPL_f, PWL_f = calculate_spl_rotor_in_frequency_domain(
                blade_number=blade_number,
                mach_number=mach_number,
                theta=theta,
                thrust=2000
            )

            SPL_t_list.append(SPL_t)
            SPL_f_list.append(SPL_f)

            PWL_t_list.append(PWL_t)
            PWL_f_list.append(PWL_f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        fig.suptitle(f'SPL and PWL for B={blade_number}, M={mach_number}')

        # plot results
        ax1.plot(np.degrees(thetas), SPL_t_list, label='time approach')
        ax1.plot(np.degrees(thetas), SPL_f_list, label='frequency approach')
        ax2.plot(np.degrees(thetas), PWL_t_list, label='time approach')
        ax2.plot(np.degrees(thetas), PWL_f_list, label='frequency approach')

        ax1.set_xlim(0, 90)
        ax1.set_ylim(0, 150)
        ax2.set_xlim(0, 90)
        ax2.set_ylim(0, 150)

        ax1.legend()
        ax2.legend()

        ax1.set_xlabel("theta [deg]")
        ax1.set_ylabel("SPL [dB]")

        ax2.set_xlabel("theta [deg]")
        ax2.set_ylabel("PWL [dB]")

        ax1.grid()
        ax2.grid()
        plt.show()
        # plt.savefig(f'./figs/directivity B{blade_number}M{int(mach_number * 10)}.png', dpi=100)
