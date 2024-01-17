from functions import *

blade_numbers = [4, 5]
tip_mach_numbers = [.3, .6]
receiver_distance = 100  # m


if __name__ == "__main__":
    for blade_number, tip_mach_number in itertools.product(blade_numbers, tip_mach_numbers):
        SPL_t_list = []
        PWL_t_list = []
        SPL_f_list = []
        PWL_f_list = []

        print(f'Running directivity plots...')
        thetas = cosspace(0, np.pi / 2, 100)
        for plot_num, theta in enumerate(thetas):
            SPL_t, PWL_t, time_receiver, p_receiver = calculate_spl_rotor_in_time_domain(
                blade_number=blade_number,
                tip_mach_number=tip_mach_number,
                theta=theta,
                thrust=2000,
                plot_pressure=False,
                harmonic_loading=False,
                receiver_distance=receiver_distance
            )

            SPL_f, PWL_f, freqs_f, PWLs_f = calculate_spl_rotor_in_frequency_domain(
                blade_number=blade_number,
                tip_mach_number=tip_mach_number,
                theta=theta,
                thrust=2000,
                harmonic_loading=False,
                receiver_distance=receiver_distance,
                phi=np.pi/3
            )

            SPL_t_list.append(SPL_t)
            SPL_f_list.append(SPL_f)

            PWL_t_list.append(PWL_t)
            PWL_f_list.append(PWL_f)


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        fig.suptitle(f'Rotor noise with B={blade_number}, M={tip_mach_number}, and constant loading')

        # plot results
        ax1.plot(np.degrees(thetas), SPL_t_list, label='time approach')
        ax1.plot(np.degrees(thetas), SPL_f_list, label='frequency approach')
        ax2.plot(np.degrees(thetas), PWL_t_list, label='time approach')
        ax2.plot(np.degrees(thetas), PWL_f_list, label='frequency approach')

        ax1.set_xlim(0, 90)
        ax1.set_ylim(0, None)
        ax2.set_xlim(0, 90)
        ax2.set_ylim(0, None)

        ax1.legend()
        ax2.legend()

        ax1.set_xlabel("theta [deg]")
        ax1.set_ylabel("SPL [dB]")

        ax2.set_xlabel("theta [deg]")
        ax2.set_ylabel("PWL [dB]")

        ax1.grid()
        ax2.grid()
        # plt.show()
        plt.savefig(f'./figs/directivity B{blade_number}M{int(tip_mach_number * 10)}const.png', dpi=100)
