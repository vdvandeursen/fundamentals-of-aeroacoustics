from functions import *


if __name__ == "__main__":

    blade_numbers = [4, 5]
    tip_mach_numbers = [.3, .6]
    receiver_distance = 100  # m

    for blade_number, tip_mach_number in itertools.product(blade_numbers, tip_mach_numbers):

        # Running PSD plots
        SPL_t, PWL_t, time_receiver, p_fluctuations = calculate_spl_rotor_in_time_domain(
            blade_number=blade_number,
            tip_mach_number=tip_mach_number,
            theta=np.radians(45),
            thrust=2000,
            plot_pressure=False,
            harmonic_loading=True,
            n_steps=2000,
            receiver_distance=receiver_distance
        )

        SPL_f, PWL_f, freqs_f, PWLs_f = calculate_spl_rotor_in_frequency_domain(
            blade_number=blade_number,
            tip_mach_number=tip_mach_number,
            theta=np.radians(45),
            thrust=2000,
            harmonic_loading=True,
            receiver_distance=receiver_distance,
            phi=0.3
        )

        fs = 1 / (time_receiver[1] - time_receiver[0])
        SPLs_t, freqs_t = PyOctaveBand.octavefilter(
            p_fluctuations,
            fs=fs,
            fraction=12
        )

        PWLs_t = np.array(SPLs_t) + 11 + 20 * np.log10(receiver_distance)

        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=100)

        ax.plot(freqs_t, PWLs_t, label='time approach')
        ax.plot(freqs_f, PWLs_f, label='freq approach')

        ax.legend()
        ax.set_ylim(0, None)
        ax.set_xlim(0, 20000)
        ax.grid()

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PWL [dBW]")
        plt.show()

