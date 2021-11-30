from fts_coupling_optics_geo import *
import fts_coupling_optics_geo as fts
import coupling_optics_sims as csims
import yaml
import pickle
import numpy as np
from multiprocessing import Process, Manager, Semaphore
import RayTraceFunctionsv2 as rt
from tqdm import tqdm
import time

c = 300.
LAST_LENS_EDGE = [-231.24377979, -266.21940725, 0.]
COUPLING_OPTICS_ORIGIN = [-233.28894593160666, -276.84436350628596, 0.]
IN_TO_MM = 1 / (csims.mm_to_in)
DET_SIZE = 5  # in mm
FTS_BEAM_ANGLE = -0.190161


def get_aspect(config, aspect, element, number):
    '''Obtain the config for a given aspect, element, and number

    Parameters:
        config (yaml file)  -- yaml configuration file loaded
        aspect (str)        -- aspect of the FTS (origins, angles, coefficients, etc)
        element (str)       -- element of the FTS for which this aspect is defined 
                              (ellipses, mirror, polarizers)
        number (int)        -- number of the element that we're specifically interested
                            in (1-10 for ellipses, 1-4 for polarizers)'''

    def get_item(dic, key, num):
        if type(dic[key]) is dict:
            return dic[key][num]
        else:
            return dic[key]

    if element is None:
        return get_item(config, aspect, number)
    else:
        return get_item(config[aspect], element, number)


def remove_outliers(out, threshold=5, debug=False):
    new_out = []
    mean_x = csims.FOCUS[0]
    mean_y = csims.FOCUS[1]
    if (debug):
        print(f'focus is at {np.array(csims.FOCUS)[[0, 1]]}')

    for i in range(out.shape[1]):
        point = out[[0, 1, 2], i]
        if np.abs(point[0] - mean_x) > threshold or np.abs(
                point[1] - mean_y) > threshold:
            if (debug):
                print(f'Taking away outlier point {point}')
            continue
        new_out.append(out[:, i])
    return np.array(new_out).T


# just subtract our z coordinate until we hit -20.9
def get_distances_at_z(out, z_coordinate):
    iters = (out[2] - z_coordinate) / out[5]
    return out[8] - iters / csims.mm_to_in


def get_rays_at_z(out, z_coordinate):
    iters = (out[2] - z_coordinate) / out[5]
    new_points = out[[0, 1, 2]] - out[[3, 4, 5]] * iters

    new_out = out.copy()
    new_out[[0, 1, 2]] = new_points
    new_out[8] = get_distances_at_z(out, z_coordinate)
    return new_out


def get_rotation_matrix(a, b):
    v = np.cross(a, b)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.identity(3) + vx + np.linalg.matrix_power(vx, 2) * (1 - c) / (
        s ** 2)


def transform_points(x_vals, y_vals, z_vals, new_origin, rotation_matrix):
    XTR = []
    YTR = []
    ZTR = []
    for i in range(0, len(x_vals)):
        v = [x_vals[i], y_vals[i], z_vals[i]]
        #v2R = rotate(v, thetaxyz)
        v2R = np.dot(v, rotation_matrix)
        v2RS = v2R + new_origin
        XTR.append(v2RS[0])
        YTR.append(v2RS[1])
        ZTR.append(v2RS[2])
    return XTR, YTR, ZTR


def transform_rays_to_coupling_optics_frame(rays):
    # we want the rays essentially directly after they hit the last polarizer
    # in the FTS and then we need to calculate the distance between this
    # polarizer and the first lens of the coupling optics

    # first we need to transform the rays' points and normal vectors to the
    # frame of the coupling optics

    # Here we really need to make sure we're properly rotating this ray really
    # the ray should stop at the plane which makes the same beam angle as the
    # coupling optics actually things should be fine I think, but just in case
    # do it this way I guess
    # don't stop at (0, 0, 0), stop at (0, -.426, 0) equivilently
    coupling_optics_origin = COUPLING_OPTICS_ORIGIN
    factor = csims.mm_to_in
    new_rays = []
    for ray in rays:
        new_ray = [ray[0], ray[1], None, None, ray[4]]
        # switch the x and z coordinate of these!
        new_vec = rt.rotate(ray[3], [0, 0, FTS_BEAM_ANGLE])
        new_ray[2] = factor * np.flip(rt.rotate(np.subtract(
            ray[2], coupling_optics_origin), [0, 0, FTS_BEAM_ANGLE]) * [
                1, -1, 1])
        new_ray[3] = np.flip(csims.normalize(factor * new_vec * [1, -1, 1]))
        new_rays.append(new_ray)

    return new_rays


def create_source_rays_uniform_from_start_displacement(
        source_origin, source_normal_vec, horiz_displacement,
        vert_displacement, n_linear_theta, n_linear_phi, config,
        check_rays=True, theta_bound=np.pi / 2, timeout=10):
    # first create rays distributed in the upwards cone
    # and then rotate them to center them around the normal
    # also create them around a variety of starting points
    # assume radially symmetric source
    rotation_matrix = get_rotation_matrix(source_normal_vec, [0, 0, 1])
    rays = []

    # n^2 computations here
    starting_time = time.time()
    for theta_val in np.linspace(0, theta_bound, n_linear_theta):
        for phi_val in np.linspace(0, 2 * np.pi, n_linear_phi):
            if time.time() - starting_time > timeout:
                print('timing out..')
                return rays

            point_origin = [horiz_displacement, vert_displacement, 0]

            # Direction of ray away from the starting point
            r_hat = [np.sin(theta_val) * np.cos(phi_val),
                     np.sin(theta_val) * np.sin(phi_val), np.cos(theta_val)]

            transformed_starting_vector = -1 * np.array(transform_points(
                [r_hat[0]], [r_hat[1]], [r_hat[2]], [0, 0, 0],
                rotation_matrix)).flatten()

            transformed_starting_point = np.array(transform_points(
                [point_origin[0]], [point_origin[1]], [point_origin[2]],
                source_origin, rotation_matrix)).flatten()

            # strategically choose our starting rays such that they make it
            # through the to the first ellipse that we hit
            polarization_angle = .123
            intensity = 1.0
            ray = [polarization_angle, intensity,
                   transformed_starting_point.tolist(),
                   transformed_starting_vector.tolist(), 0]
            # paths = ['OM2', 'A1', 'OM1', 'T4', 'E6']
            if (check_rays):
                paths = ['T4', 'E6']
                final_ray = rt.run_ray_through_sim(ray, config, None, paths)
                if (final_ray is not None):
                    rays.append(ray)
            else:
                rays.append(ray)

    return rays


def get_centers(n_linear_det):
    displacements = []
    centers = np.linspace(-(n_linear_det // 2), (n_linear_det // 2),
                          n_linear_det) * DET_SIZE
    for x_center in centers:
        for y_center in centers:
            displacements.append((x_center, y_center))
    return displacements


def segment_detector(outrays, n_linear_det=5):
    points = outrays[[0, 1, 2]] * IN_TO_MM

    # divide these into segments of length 5mm
    centers = np.linspace(-(n_linear_det // 2), (n_linear_det // 2),
                          n_linear_det) * DET_SIZE
    out_data = []
    point_data = []
    for x_center in centers:
        for y_center in centers:
            center = (x_center, y_center + csims.FOCUS[1] * IN_TO_MM)
            # get all the rays that go into this detector
            distance_from_x = np.square(points[0] - center[0])
            distance_from_y = np.square(points[1] - center[1])
            dist_from_center = np.sqrt(distance_from_x + distance_from_y)
            segment_within = np.where(dist_from_center < (DET_SIZE / 2))[0]
            out_within = outrays[:, segment_within]

            out_data.append(out_within)
            point_data.append([x_center, y_center])
    return out_data, np.array(point_data)


def trace_rays(start_displacement, n_mirror_positions, ymax,
               n_linear_theta=100, n_linear_phi=100, debug=False):
    # For each starting position, trace the rays and segment the detectors.
    # Create an interferogram for each detector for each starting position
    # i.e. each starting position should have 36 interferograms (1 raytrace
    # iter).
    with open("lab_fts_dims_act.yml", "r") as stream:
        config = yaml.safe_load(stream)

    starting_rays = create_source_rays_uniform_from_start_displacement(
        config['detector']['center'], config['detector']['normal_vec'],
        start_displacement[0], start_displacement[1], n_linear_theta,
        n_linear_phi, config, theta_bound=np.pi / 2, check_rays=True,
        timeout=100)

    with open("lab_fts_dims_mcmahon_backwards.yml", "r") as stream:
        config = yaml.safe_load(stream)

    possible_paths = [path[::-1] for path in rt.get_possible_paths()]
    delay, final_rays = rt.run_all_rays_through_sim_optimized(
        starting_rays, config, n_mirror_positions, paths=possible_paths,
        ymax=ymax, progressbar=(debug))

    total_outrays = []

    for rays in tqdm(final_rays, disable=(not debug)):
        transformed_rays = transform_rays_to_coupling_optics_frame(rays)
        out_forwards = csims.run_rays_forwards_input_rays(
            transformed_rays, z_ap=csims.FOCUS[2], plot=False)
        total_outrays.append(out_forwards)

    return total_outrays


# remove the outliers for this as well!
def segment_rays(total_out, n_linear_det=5):
    total_out_segments = []
    for out in total_out:
        out_segments, det_points = segment_detector(
            remove_outliers(out), n_linear_det=n_linear_det)
        total_out_segments.append(out_segments)
    return total_out_segments, det_points


def data_to_matrix(center_data):
    max_rays_num = max([data.shape[1] for data in center_data])
    if max_rays_num == 0:
        return None
    total_rays = []
    for data in center_data:
        # only keep polarization, intensity, and phase
        # could also keep detector position and angle separately
        # if we're curious!
        power_vals_mask = [6, 7, 8]
        power_vals = data[power_vals_mask]
        power_vals = power_vals.T
        # pad rows now
        total_vals = np.zeros((max_rays_num, 3))
        total_vals[:power_vals.shape[0]] = power_vals
        total_rays.append(total_vals)

    return np.array(total_rays)


def get_interferogram_frequency(outrays, frequencies, debug=True):
    theta = outrays[:, :, 0]
    intensity = outrays[:, :, 1]
    distance = outrays[:, :, 2]
    ex1 = np.sqrt(intensity) * np.cos(theta)
    ey1 = np.sqrt(intensity) * np.sin(theta)

    total_power = np.zeros(outrays.shape[0])
    for freq in tqdm(frequencies, disable=(not debug)):
        wavelength = c / freq
        phase = np.exp(1j * (distance * 2 * np.pi / wavelength))
        ex = ex1 * phase
        ey = ey1 * phase

        power = np.square(np.abs((ex.sum(axis=1)))) + np.square(
            np.abs((ey.sum(axis=1))))
        total_power += power
    return total_power


def get_interferograms(out_data, freqs):
    total_out_segments, det_points = segment_rays(out_data)
    reorganized_segments = []
    for j in range(len(det_points)):
        reorganized_segments.append([])
    for i in range(len(total_out_segments)):
        for j in range(len(total_out_segments[i])):
            data = total_out_segments[i][j]
            # if data is empty, do nothing
            reorganized_segments[j].append(data)

    interferograms = []
    for j, det_center in enumerate(det_points):
        # get the max number of rays
        data_matrix = data_to_matrix(reorganized_segments[j])
        if (data_matrix is None):
            interferogram = np.zeros(len(reorganized_segments[0]))
        else:
            interferogram = get_interferogram_frequency(
                data_matrix, freqs, debug=False)
            #interferogram = rt.get_interferogram(data_matrix, (c / 150.))
        interferograms.append(interferogram)
    return interferograms


def get_and_combine_interferograms(all_data, freqs, debug=True):
    total_interferograms = []
    for data in tqdm(all_data, disable=(not debug)):
        interferograms = get_interferograms(data, freqs)
        total_interferograms.append(interferograms)
    total_interferograms = np.array(total_interferograms)

    print(total_interferograms.shape)
    return total_interferograms


def add_outrays(total_outrays, start_displacement, n_mirror_positions,
                y_max, n_linear_theta, n_linear_phi, semaphore, debug):
    outrays = trace_rays(start_displacement, n_mirror_positions, y_max,
                         n_linear_theta=n_linear_theta,
                         n_linear_phi=n_linear_phi, debug=debug)
    total_outrays.put(outrays)
    semaphore.release()


def get_outrays_threaded(
        x_displacements, y_displacements, n_mirror_positions, y_max,
        n_linear_theta=50, n_linear_phi=50, debug=False):
    processes = []
    max_processes = 55
    semaphore = Semaphore(max_processes)
    manager = Manager()
    total_outrays = manager.Queue()

    for x in x_displacements:
        for y in y_displacements:
            start_displacement = (x, y)
            semaphore.acquire()
            process = Process(target=add_outrays, args=(
                total_outrays, start_displacement, n_mirror_positions,
                y_max, n_linear_theta, n_linear_phi, semaphore, debug),
                daemon=True)

            print('process %s starting.' % len(processes))
            process.start()
            processes.append(process)

    # Wait for all the processes to finish before converting to a list.
    for i, process in enumerate(processes):
        print('process %s joining.' % i)
        process.join()
        print('process %s finishg.' % i)

    # convert the shared queue to a list.
    outrays_list = []
    while (total_outrays.qsize() != 0):
        outrays_list.append(total_outrays.get())
    return outrays_list


def main():
    # FTS_stage_step_size = 0.375  # FTS step size in mm
    # FTS_stage_step_size = 0.15  # FTS step size in mm
    # FTS_stage_throw = 35.     # total throw extent in mm

    # n_mirror_positions = (2 * FTS_stage_throw / FTS_stage_step_size)
    # n_mirror_positions = 2 * (FTS_stage_throw / .15)

    FTS_stage_throw = 20.     # total throw extent in mm
    FTS_stage_step_size = 0.1  # FTS step size in mm
    # n_mirror_positions = (20 * 2 / .1)
    n_mirror_positions = (2 * FTS_stage_throw / FTS_stage_step_size)

    x_vals = np.linspace(0, 0, 2)
    y_vals = np.linspace(0, 0, 1)
    #z_vals = np.linspace(0, 0, 1)

    total_outrays_0 = get_outrays_threaded(
        x_vals, y_vals, n_mirror_positions, FTS_stage_throw,
        n_linear_theta=20, n_linear_phi=20, debug=True)

    pickle.dump(total_outrays_0, open(
        "data/total_outrays_0_test.p", "wb"))

    # save this for loading elsewhere
    print('finished!')


if __name__ == '__main__':
    main()
    exit()
