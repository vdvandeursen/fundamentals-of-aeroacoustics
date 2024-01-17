from functions import *
import open3d as o3d
import pandas as pd
import itertools
import tqdm

if __name__ == "__main__":
    scale_factor = 1
    n_x = 50
    n_y = 50
    n_z = 15

    x_coordinates = scale_factor * np.linspace(-400, 600, n_x)
    y_coordinates = scale_factor * np.linspace(-500, 500, n_y)
    z_coordinates = scale_factor * np.linspace(-40, 320, n_z)

    X, Y, Z = np.meshgrid(x_coordinates, y_coordinates, z_coordinates)

    rotor_hub = np.array([100, 50, 150])  # xyz

    coordinates = np.array([X.flatten(),
                            Y.flatten(),
                            Z.flatten()]).transpose()

    distances = np.linalg.norm(coordinates - rotor_hub, axis=1)
    thetas = np.arccos((coordinates[:, 2] - 150) / distances)

    # frequency_approach = np.vectorize(calculate_spl_rotor_in_frequency_domain,
    #                                   excluded=['blade_number', 'tip_mach_number', 'thrust', 'harmonic_loading', 'return_spl_only', 'phi'])

    res = []

    x0, y0, z0 = 100, 50, 150

    for x, y, z in tqdm.tqdm(itertools.product(x_coordinates, y_coordinates, z_coordinates), total=n_x*n_y*n_z):
        dist = np.sqrt(((x0 - x)**2 + (y0 - y)**2 + (z0 - z)**2))
        theta = np.arccos((z0 - z) / dist)

        SPL = calculate_spl_rotor_in_frequency_domain(
            theta=theta,
            receiver_distance=dist,
            blade_number=4,
            tip_mach_number=0.6,
            thrust=2000,
            harmonic_loading=False,
            return_spl_only=True,
            phi=np.pi/6
        )

        res.append({
            'x': x,
            'y': y,
            'z': z,
            'r': dist,
            'theta': np.degrees(theta),
            'SPL [dB]': np.real(SPL)
        })

    # SPLs = np.real(np.array(SPLs))
    # SPLs = np.real(SPLs)

    # df = pd.DataFrame(coordinates/scale_factor, columns=['x', 'y', 'z'])
    # df.loc[:, 'SPL [dB]'] = SPLs

    df = pd.DataFrame(res)

    df.to_csv('spl_data_paris.csv', index=False)

    print('break')

