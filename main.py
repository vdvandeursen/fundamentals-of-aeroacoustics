import numpy as np
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

thetas = np.radians([0, 15, 30, 45, 60, 75, 90])  # radians
receiver_distance = 100  # m


def calculate_pressure_dipole(t, omega, receiver_distance, phi, force_magnitude):
    receiver_position = receiver_distance * np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])  # Cartesian (x,y,z)
    force_position = force_radius * np.array([np.cos(omega * t), np.sin(omega * t), 0])  # Cartesian (x,y,z)

    r = receiver_position - force_position
    r_hat = r / np.linalg.norm(r)

    F = force_magnitude * np.array(
        [-np.sin(gamma) * np.sin(omega * t), np.sin(gamma) * np.cos(omega * t), np.cos(gamma)])
    M = mach_number * np.array([-np.sin(omega * t), np.cos(omega * t), 0])

    F_dot = force_magnitude * np.array(
        [-omega * np.sin(gamma) * np.cos(omega * t), -omega * np.sin(gamma) * np.sin(omega * t), 0])
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


def calculate_spl_rotor(blade_number, mach_number, theta, thrust, show, ax=None):
    omega = mach_number * c / force_radius  # Hz
    n_steps = 1000

    if ax is None:
        _, ax = plt.subplots(1)

    time_domain_approach = np.vectorize(calculate_pressure_dipole, excluded=['omega', 'receiver_distance', 'phi', 'force_magnitude'])

    blade_receiver_times = []
    p_blades = []

    # Calculate pressure contribution for each blade in time domain approach
    for B in range(0, blade_number):
        emission_time = np.linspace(0, 4 * np.pi / omega, n_steps) # simulate each dipole 2 full rotations
        travel_time_blade, p_blade = time_domain_approach(
            emission_time,
            omega=omega,
            phi=B*2*np.pi/blade_number,
            receiver_distance=receiver_distance,
            force_magnitude=thrust/blade_number
        )
        receiver_time_blade = emission_time + travel_time_blade

        blade_receiver_times.append(receiver_time_blade)
        p_blades.append(p_blade)

    # sample dipole pressure contributions between 0.5 and 1.5 rotor rotations
    time_receiver_lower_bound = np.pi / omega + receiver_distance / c
    time_receiver_upper_bound = 3 * np.pi / omega + receiver_distance / c
    time_receiver = np.linspace(time_receiver_lower_bound, time_receiver_upper_bound, n_steps)
    p_receiver = 0

    for i, (receiver_time_blade, p_blade) in enumerate(zip(blade_receiver_times, p_blades)):
        p = np.interp(
            x=time_receiver,
            xp=receiver_time_blade,
            fp=p_blade
        )
        p_receiver += p

        ax.plot(
            time_receiver,
            p,
            label=f'Blade {i+1}',
            linestyle='dotted'
        )

    p_rms = np.sqrt(np.mean(p_receiver**2))
    SPL = 10*np.log10(p_rms**2 / p_ref**2)

    I = p_rms**2 / (rho0 * c)
    P = I * receiver_distance**2 * 4*np.pi

    PWL = 10*np.log10(P/P_ref)

    # Create plots
    ax.plot(
        time_receiver,
        p_receiver,
        label=f'Receiver',
        linewidth=2
    )

    ax.set_title(rf'$\theta$={np.degrees(theta):.2f}$^\circ$')
    ax.legend()
    ax.set_xlabel("Receiver time [s]")
    ax.set_ylabel("Pressure [Pa]")

    ax.set_ylim(-0.1, 0.1)

    if show:
        plt.show()

    return SPL, PWL


if __name__ == "__main__":

    for blade_number, mach_number in itertools.product(blade_numbers, mach_numbers):
        fig, axs = plt.subplots(int(np.ceil(len(thetas)/2)), 2, figsize=(19, 10))

        fig.suptitle(f'SPL for B={blade_number}, M={mach_number}')

        for plot_num, theta in enumerate(thetas):
            SPL, PWL = calculate_spl_rotor(
                blade_number=blade_number,
                mach_number=mach_number,
                theta=theta,
                thrust=2000,
                show=False,
                ax=axs[int(np.floor(plot_num/2)), int(np.ceil(plot_num % 2))]
            )
        # plt.tight_layout()

            print(f'B={blade_number} M={mach_number}, theta={np.degrees(theta):.2f} gives     SPL={SPL:.2f}dB, PWL={PWL:.2f}dB')

        plt.savefig(f'./figs/SPL B{blade_number}M{int(mach_number*10)}.png', dpi=100)
        print('break')
