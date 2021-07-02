# Code largely written by Grace Chesmore with a few modifications, mostly for
# plotting, by Tommy Alford.

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial import distance
import fts_coupling_optics_geo as fts
from fts_coupling_optics_geo import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import random
#pio.renderers.default = 'notebook'

# %matplotlib inline

# plt.style.use('ggplot')

# font = {'family': 'Arial',
#         'size': 15}
# plt.rc('font', **font)

# %load_ext blackcellmagic


def snell_vec(n1, n2, N_surf, s1):
    # s1 is the incoming vector, pointing from the light source to the surface
    # N_surf is the normal of the surface

    s2 = (n1/n2) * np.cross(N_surf, (np.cross(-N_surf, s1))) \
        - N_surf*np.sqrt(1-(n1/n2)**2 * np.dot((np.cross(N_surf, s1)),
                                               (np.cross(N_surf, s1))))
    return s2


x = np.linspace(-(3), (3), 100)  # [in]
y = np.linspace(-(3), (3), 100)  # [in]
X, Y = np.meshgrid(x, y)

# Define all surfaces of lenses
Z_l1s1 = l1s1(X, Y)
Z_l1s2 = l1s2(X, Y)

Z_l2s1 = l2s1(X, Y)
Z_l2s2 = l2s2(X, Y)

Z_l3s1 = l3s1(X, Y)
Z_l3s2 = l3s2(X, Y)

Z_flat = flat(X, Y)

# These functions rotate a given surface and shift it to its appropriate location
# in the FTS coupling optics setup. I gave an  option to shift, or not shift, because
# if you want to switch references frames of a normal vector, you do NOT want to shift it,
# you just want to rotate it.  Here I shift it because i want to visualize it in the
# complete setup:
shift = 1
Xt11, Yt11, Zt11 = ref_11_to_tele(X, Y, Z_l1s1, shift)
Xt12, Yt12, Zt12 = ref_12_to_tele(X, Y, Z_l1s2, shift)

Xt21, Yt21, Zt21 = ref_21_to_tele(X, Y, Z_l2s1, shift)
Xt22, Yt22, Zt22 = ref_22_to_tele(X, Y, Z_l2s2, shift)

Xtflat, Ytflat, Ztflat = ref_flat_to_tele(X, Y, Z_flat, shift)
Xt31, Yt31, Zt31 = ref_31_to_tele(X, Y, Z_l3s1, shift)
Xt32, Yt32, Zt32 = ref_32_to_tele(X, Y, Z_l3s2, shift)


def plot_surface(x1, y1, z1, x2, y2, z2, fig):
    x1_tot, y1_tot, z1_tot = ([], [], [])
    x2_tot, y2_tot, z2_tot = ([], [], [])

    for l in range(0, 100, 1):
        x1_tot.extend(x1[:, l])
        y1_tot.extend(y1[:, l])
        z1_tot.extend(z1[:, l])

        x2_tot.extend(x2[:, l])
        y2_tot.extend(y2[:, l])
        z2_tot.extend(z2[:, l])

    #l = int(len(y1) / 2)
    line1 = go.Mesh3d(x=x1_tot, y=y1_tot, z=z1_tot, color='blue', opacity=.4)
    line2 = go.Mesh3d(x=x2_tot, y=y2_tot, z=z2_tot, color='blue', opacity=.4)
    fig.add_trace(line1)
    fig.add_trace(line2)


def plot_surface(x, y, z, fig, start, stop, color, half_index=50):
    x_tot, y_tot, z_tot1, x_tot2, y_tot2, z_tot2 = ([], [], [], [], [], [])

    for l in range(start, stop, 1):
        x_tot.extend(x[:half_index, l])
        y_tot.extend(y[:half_index, l])
        z_tot1.extend(z[:half_index, l])

        x_tot2.extend(x[half_index:, l])
        y_tot2.extend(y[half_index:, l])
        z_tot2.extend(z[half_index:, l])

    # Change x and z directions to match FTS defined
    shape = go.Mesh3d(x=x_tot, y=y_tot, z=z_tot1, color=color, opacity=.4)
    shape2 = go.Mesh3d(x=x_tot2, y=y_tot2, z=z_tot2, color=color, opacity=.4)
    fig.add_trace(shape)
    fig.add_trace(shape2)


def plot_ray(x_vals, y_vals, z_vals, fig, color, alpha):
    line = go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', showlegend=False,
                        line=dict(color=color), opacity=alpha)
    fig.add_trace(line)


def norm(v):
    return np.sqrt(np.sum(np.square(v)))


def normalize(v):
    return v / norm(v)


def convert_to_ray(out_arr):
    out_point = out_arr[[0, 1, 2]]
    out_vector = out_arr[[8, 9, 10]]
    distance = out_arr[3]

    xz_factor = 17.9643
    y_factor = 14.0897

    # switch the x and z coordinate of these!
    out_point = np.flip(out_point * [xz_factor, 0, xz_factor])
    out_vector = np.flip(
        normalize(out_vector * [-1 * xz_factor, -1 * y_factor, xz_factor]))

    # need values for polarization, intensity
    polarization_angle = random.random() * 2 * np.pi
    intensity = 1.0

    ray = [polarization_angle, intensity, out_point, out_vector,
           distance * np.sqrt(xz_factor ** 2 + y_factor ** 2)]
    return ray


def convert_to_ray_mcmahon(out_arr):
    out_point = out_arr[[0, 1, 2]]
    out_vector = out_arr[[8, 9, 10]]
    distance = out_arr[3]

    # we need to convert from mm to inches here!
    factor = (1 / mm_to_in)

    # switch the x and z coordinate of these!
    out_point = factor * np.flip(out_point * [1, 0, 1])
    out_vector = np.flip(
        normalize(factor * out_vector * [1, -1, 1]))

    # need values for polarization, intensity
    polarization_angle = random.random() * 2 * np.pi
    intensity = 1.0

    ray = [polarization_angle, intensity, out_point, out_vector,
           distance * factor]
    return ray


# add a loss tangent to this index of refraction
n_lens = 1.517 + 1e-6 * np.complex(0, 1)  # index of refraction for HDPE
n_lens = 1.517
n_vac = 1       # index of refraction vacuum


def aperature_fields(P_rx, tele_geo, col, fig, n_linear=40, alpha=.04,
                     theta_bound=.1, plot=True, y_ap=-2.34):

    # y_ap = -2.34 # this is where you want your rays to end (in y). You can definitely change this! I was just guessing.
    #y_ap = -20.9
    N_linear = tele_geo.N_scan  # number of rays in 1D scan
    N_linear = n_linear
    alph = 0.04  # transparency of plotted lines
    alph = alpha

    # Step 1:  grid the plane of rays shooting out of FTS
    # In your code, you would just input the direction of a given ray
    # and probably skip this step. I'm just creating a very tidy array
    # of rays but you will already have that leaving the FTS.
    #theta_bound = .1
    theta = np.linspace((0) - theta_bound, (0) + theta_bound, N_linear)
    phi = np.linspace((0) - (np.pi/2), (0) + (np.pi/2), N_linear)

    theta, phi = np.meshgrid(theta, phi)
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    # Step 2: calculate the position + local surface normal for the dish
    n_pts = len(theta)
    #out = np.zeros((17, n_pts))
    out = [[], [], [], [], [], [], [], [], [], [], []]

    for ii in range(n_pts):

        th = theta[ii]
        ph = phi[ii]

        # Direction of ray away from the starting point
        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # I refer to the starting point of the rays as 'rx' throughout the code.
        # So when you see that, that's what that means.
        x_rx = P_rx[0]
        y_rx = P_rx[1]
        z_rx = P_rx[2]

        # To hit a surface, I use a root finder. This tells the ray to search
        # for a distance 't' steps away until it hits a defined surface. The variable
        # 't' is basically how many normal vectors do I need to travel before I hit
        # the desired surface. I do this for every surface throughout this function.

        def root_32(t):
            x = x_rx + alpha * t
            y = y_rx + beta * t
            z = z_rx + gamma * t
            x32, y32, z32 = ref_tele_to_32(
                x, y, z, 1
            )
            z_32 = l3s2(x32, y32)
            if np.isnan(z_32) == True:
                z_32 = 0
            root = z32 - z_32
            return root

        try:
            t_32 = optimize.brentq(root_32, 0.01, 500)  # [in.]
        except ValueError:
            # print(ii)
            continue
        # f(a) and f(b) need to be opposite signs

        # Location of where ray hits M1
        x_32 = P_rx[0] + alpha * t_32
        y_32 = P_rx[1] + beta * t_32
        z_32 = P_rx[2] + gamma * t_32
        P_32 = np.array([x_32, y_32, z_32])

        ###### in lens surface reference frame ##########################
        x_32_temp, y_32_temp, z_32_temp = ref_tele_to_32(
            x_32, y_32, z_32, 1
        )  # P_32 temp
        x_rx_temp, y_rx_temp, z_rx_temp = ref_tele_to_32(
            x_rx, y_rx, z_rx, 1
        )  # P_rx temp
        norm = d_z(x_32_temp, y_32_temp, lens3_surf2, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_rx_32 = np.array([x_32_temp, y_32_temp, z_32_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_32 = np.sqrt(np.sum(vec_rx_32 ** 2))
        tan_rx_32 = vec_rx_32 / dist_rx_32

        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_rx_32)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_32_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_rx_32_t_x, tan_rx_32_t_y, tan_rx_32_t_z = ref_32_to_tele(
            tan_rx_32[0], tan_rx_32[1], tan_rx_32[2], 0
        )
        tan_rx_32_t = [tan_rx_32_t_x, tan_rx_32_t_y, tan_rx_32_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_32_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]  # outgoing ray vector

        # I've commented these out, but if you want to view the outgoing ray direction,
        # normal vector of surface, incoming ray direction, and  of this surface
        # you can uncomment and plot the normal vector of the 13th ray.

        # if ii==13:
        #     plt.plot([y_32,y_32+(5*tan_og_t[1])], [z_32,z_32+(5*tan_og_t[2])], "-",color = 'r')
        #     plt.plot([y_32,y_32+(5*N_hat_t[1])], [z_32,z_32+(5*N_hat_t[2])], "-",color = 'k')
        #     plt.plot([y_32,y_32-(5*tan_rx_32_t[1])], [z_32,z_32-(5*tan_rx_32_t[2])], "-",color = 'g')

        ################
        ################
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_31(t):
            x = x_32 + alpha * t
            y = y_32 + beta * t
            z = z_32 + gamma * t
            x31, y31, z31 = ref_tele_to_31(
                x, y, z, 1
            )
            z_31 = l3s1(x31, y31)
            if np.isnan(z_31) == True:
                z_31 = 0
            root = z31 - z_31
            return root
        try:
            t_31 = optimize.brentq(root_31, 0.01, 500)
        except ValueError:
            # print(ii)
            continue
        # Location of where ray hits M1
        x_31 = P_32[0] + alpha * t_31
        y_31 = P_32[1] + beta * t_31
        z_31 = P_32[2] + gamma * t_31
        P_31 = np.array([x_31, y_31, z_31])

        ###### in lens surface reference frame ##########################
        x_31_temp, y_31_temp, z_31_temp = ref_tele_to_31(
            x_31, y_31, z_31, 1
        )  # P_m2 temp
        x_32_temp, y_32_temp, z_32_temp = ref_tele_to_31(
            x_32, y_32, z_32, 1
        )  # P_rx temp
        norm = d_z(x_31_temp, y_31_temp, lens3_surf1, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])

        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_32_31 = np.array([x_31_temp, y_31_temp, z_31_temp]) - np.array(
            [x_32_temp, y_32_temp, z_32_temp]
        )
        dist_32_31 = np.sqrt(np.sum(vec_32_31 ** 2))
        tan_32_31 = vec_32_31 / dist_32_31

        tan_og_lens = snell_vec(n_lens, n_vac, -N_hat, tan_32_31)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_31_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_32_31_t_x, tan_32_31_t_y, tan_32_31_t_z = ref_31_to_tele(
            tan_32_31[0], tan_32_31[1], tan_32_31[2], 0
        )
        tan_32_31_t = [tan_32_31_t_x, tan_32_31_t_y, tan_32_31_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_31_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]
        ##################################################

        def root_flat(t):

            x = x_31 + alpha * t
            y = y_31 + beta * t
            z = z_31 + gamma * t
            xflat, yflat, zflat = ref_tele_to_flat(
                x, y, z, 1
            )
            z_flat = flat(xflat, yflat)
            if np.isnan(z_flat) == True:
                z_flat = 0
            root = zflat - z_flat
            return root

        try:
            t_flat = optimize.brentq(root_flat, 0.03, 50)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits M2
        x_flat = x_31 + alpha * t_flat
        y_flat = y_31 + beta * t_flat
        z_flat = z_31 + gamma * t_flat
        P_flat = np.array([x_flat, y_flat, z_flat])

        ###### in lens surface reference frame ##########################
        x_flat_temp, y_flat_temp, z_flat_temp = ref_tele_to_flat(
            x_flat, y_flat, z_flat, 1
        )  # P_m2 temp
        x_31_temp, y_31_temp, z_31_temp = ref_tele_to_flat(
            x_31, y_31, z_31, 1
        )  # P_rx temp
        norm_temp = np.array([0, 0, 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_31_flat = np.array([x_flat_temp, y_flat_temp, z_flat_temp]) - np.array(
            [x_31_temp, y_31_temp, z_31_temp]
        )
        dist_31_flat = np.sqrt(np.sum(vec_31_flat ** 2))
        tan_31_flat = vec_31_flat / dist_31_flat

        tan_og_lens = 2 * (np.dot(N_hat, - tan_31_flat)
                           * (N_hat)) - (-tan_31_flat)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_flat_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_31_flat_t_x, tan_31_flat_t_y, tan_31_flat_t_z = ref_flat_to_tele(
            tan_31_flat[0], tan_31_flat[1], tan_31_flat[2], 0
        )
        tan_31_flat_t = [tan_31_flat_t_x, tan_31_flat_t_y, tan_31_flat_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_flat_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_22(t):
            x = P_flat[0] + alpha * t
            y = P_flat[1] + beta * t
            z = P_flat[2] + gamma * t

            x22, y22, z22 = ref_tele_to_22(
                x, y, z, 1
            )
            z_22 = l2s2(x22, y22)
            if np.isnan(z_22) == True:
                z_22 = 0
            root = z22 - z_22
            return root

        try:
            t_22 = optimize.brentq(root_22, 0.05, 10)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits lens surface
        x_22 = P_flat[0] + alpha * t_22
        y_22 = P_flat[1] + beta * t_22
        z_22 = P_flat[2] + gamma * t_22
        P_22 = np.array([x_22, y_22, z_22])

        ###### in lens surface reference frame ##########################
        x_22_temp, y_22_temp, z_22_temp = ref_tele_to_22(
            x_22, y_22, z_22, 1
        )  # P_m2 temp
        x_flat_temp, y_flat_temp, z_flat_temp = ref_tele_to_22(
            x_flat, y_flat, z_flat, 1
        )  # P_rx temp
        norm = d_z(x_22_temp, y_22_temp, lens2_surf2, "spheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_flat_22 = np.array([x_22_temp, y_22_temp, z_22_temp]) - np.array(
            [x_flat_temp, y_flat_temp, z_flat_temp]
        )
        dist_flat_22 = np.sqrt(np.sum(vec_flat_22 ** 2))
        tan_flat_22 = vec_flat_22 / dist_flat_22

        # Use Snell's Law to find angle of outgoing ray:
        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_flat_22)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_22_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_flat_22_t_x, tan_flat_22_t_y, tan_flat_22_t_z = ref_22_to_tele(
            tan_flat_22[0], tan_flat_22[1], tan_flat_22[2], 0
        )
        tan_flat_22_t = [tan_flat_22_t_x, tan_flat_22_t_y, tan_flat_22_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_22_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        ##################################################
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_21(t):
            x = P_22[0] + alpha * t
            y = P_22[1] + beta * t
            z = P_22[2] + gamma * t

            x21, y21, z21 = ref_tele_to_21(
                x, y, z, 1
            )  # take ray end coordinates and convert to M1 coordinates
            z_21 = l2s1(x21, y21)  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_21) == True:
                z_21 = 0
            root = z21 - z_21
            return root

        try:
            t_21 = optimize.brentq(root_21, .01, 30)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits M1
        x_21 = P_22[0] + alpha * t_21
        y_21 = P_22[1] + beta * t_21
        z_21 = P_22[2] + gamma * t_21
        P_21 = np.array([x_21, y_21, z_21])

        ###### in lens surface reference frame ##########################
        x_21_temp, y_21_temp, z_21_temp = ref_tele_to_21(
            x_21, y_21, z_21, 1
        )  # P_m2 temp
        x_22_temp, y_22_temp, z_22_temp = ref_tele_to_21(
            x_22, y_22, z_22, 1
        )  # P_rx temp
        norm = d_z(x_21_temp, y_21_temp, lens2_surf1, "spheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_22_21 = np.array([x_21_temp, y_21_temp, z_21_temp]) - np.array(
            [x_22_temp, y_22_temp, z_22_temp]
        )
        dist_22_21 = np.sqrt(np.sum(vec_22_21 ** 2))
        tan_22_21 = vec_22_21 / dist_22_21

        # Use Snell's Law to find angle of outgoing ray:

        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_22_21)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_21_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_22_21_t_x, tan_22_21_t_y, tan_22_21_t_z = ref_21_to_tele(
            tan_22_21[0], tan_22_21[1], tan_22_21[2], 0
        )
        tan_22_21_t = [tan_22_21_t_x, tan_22_21_t_y, tan_22_21_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_21_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_12(t):

            x = x_21 + alpha * t
            y = y_21 + beta * t
            z = z_21 + gamma * t

            x12, y12, z12 = ref_tele_to_12(
                x, y, z, 1
            )  # Convert ray's endpoint into M2 coordinates

            z_12 = l1s2(x12, y12)  # Z of mirror in M2 coordinates
            if np.isnan(z_12) == True:
                z_12 = 0
            root = z12 - z_12
            return root

        try:
            t_12 = optimize.brentq(root_12, 1e-3, 90)
        except ValueError:
            # print(ii)
            continue

        x_12 = P_21[0] + alpha * t_12
        y_12 = P_21[1] + beta * t_12
        z_12 = P_21[2] + gamma * t_12
        P_12 = np.array([x_12, y_12, z_12])

        ###### in lens surface reference frame ##########################
        x_12_temp, y_12_temp, z_12_temp = ref_tele_to_12(
            x_12, y_12, z_12, 1
        )  # P_m2 temp
        x_21_temp, y_21_temp, z_21_temp = ref_tele_to_12(
            x_21, y_21, z_21, 1
        )  # P_rx temp
        norm = d_z(x_12_temp, y_12_temp, lens1_surf2, "aspheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_21_12 = np.array([x_12_temp, y_12_temp, z_12_temp]) - np.array(
            [x_21_temp, y_21_temp, z_21_temp]
        )
        dist_21_12 = np.sqrt(np.sum(vec_21_12 ** 2))
        tan_21_12 = vec_21_12 / dist_21_12

        tan_og_lens = snell_vec(n_vac, n_lens, N_hat, tan_21_12)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_12_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [N_hat_t_x, N_hat_t_y, N_hat_t_z]
        tan_21_12_t_x, tan_21_12_t_y, tan_21_12_t_z = ref_12_to_tele(
            tan_21_12[0], tan_21_12[1], tan_21_12[2], 0
        )
        tan_21_12_t = [tan_21_12_t_x, tan_21_12_t_y, tan_21_12_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_12_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_11(t):

            x = x_12 + alpha * t  # the variable 't' I explained above ^
            y = y_12 + beta * t
            z = z_12 + gamma * t

            x11, y11, z11 = ref_tele_to_11(
                x, y, z, 1
            )  # Convert ray's endpoint into the surface's reference frame

            z_11 = l1s1(x11, y11)  # Z of surface in surface's reference frame
            # I just put this in here to make sure nothing crazy happens (i.e. avoid nan values)
            if np.isnan(z_11) == True:
                z_11 = 0
            root = z11 - z_11
            return root

        try:
            t_11 = optimize.brentq(root_11, 1e-3, 90)
        except ValueError:
            # print(ii)
            continue

        # Location of where ray hits lens 1 surface 1
        x_11 = x_12 + alpha * t_11
        y_11 = y_12 + beta * t_11
        z_11 = z_12 + gamma * t_11
        P_11 = np.array([x_11, y_11, z_11])

        ###### Transform into lens 1 surface 1 coordinates ##########################
        x_11_temp, y_11_temp, z_11_temp = ref_tele_to_11(
            x_11, y_11, z_11, 1
        )
        x_12_temp, y_12_temp, z_12_temp = ref_tele_to_11(
            x_12, y_12, z_12, 1)  # P_rx temp
        norm = d_z(x_11_temp, y_11_temp, lens1_surf1, "spheric")
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_12_11 = np.array([x_11_temp, y_11_temp, z_11_temp]) - np.array(
            [x_12_temp, y_12_temp, z_12_temp]
        )
        dist_12_11 = np.sqrt(np.sum(vec_12_11 ** 2))
        tan_12_11 = vec_12_11 / dist_12_11

        # Use Snell's Law to find angle of outgoing ray:
        tan_og_lens = snell_vec(n_lens, n_vac, N_hat, tan_12_11)

        ###### Transform back into telescope reference frame ##########################
        N_hat_t_x, N_hat_t_y, N_hat_t_z = ref_11_to_tele(
            N_hat[0], N_hat[1], N_hat[2], 0
        )
        N_hat_t = [-N_hat_t_x, -N_hat_t_y, -N_hat_t_z]
        tan_12_11_t_x, tan_12_11_t_y, tan_12_11_t_z = ref_11_to_tele(
            tan_12_11[0], tan_12_11[1], tan_12_11[2], 0
        )
        tan_12_11_t = [tan_12_11_t_x, tan_12_11_t_y, tan_12_11_t_z]
        tan_og_t_x, tan_og_t_y, tan_og_t_z = ref_11_to_tele(
            tan_og_lens[0], tan_og_lens[1], tan_og_lens[2], 0
        )
        tan_og_t = [tan_og_t_x, tan_og_t_y, tan_og_t_z]

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        dist_11_ap = abs((y_ap - P_11[1]) / tan_og_t[1])

        total_path_length = (
            t_11*n_lens + t_12 + t_21*n_lens + t_22 +
            t_flat + t_31*n_lens + t_32 + dist_11_ap
        )
        pos_ap = P_11 + dist_11_ap * np.array(tan_og_t)

        # Estimate theta, phi of outgoing initial ray
        de_ve = np.arctan(tan_rx_32_t[2] / (-tan_rx_32_t[1]))
        de_ho = np.arctan(
            tan_rx_32_t[0] / np.sqrt(tan_rx_32_t[2] ** 2 + tan_rx_32_t[1] ** 2)
        )

        #################################################
        ## Plotting lenses and rays #####################
        # if np.mod(ii, 1) == 0:
        if (ii == 0):
            # if ii == 21:
            if True:
                pass
#                 plt.plot(
#                     Yt11[:, int(len(Yt11) / 2)],
#                     Zt11[:, int(len(Yt11) / 2)],
#                     "-",
#                     color="k",
#                     label="L1S1",
#                 )

#                 plt.plot(
#                     Yt12[:, int(len(Yt12) / 2)],
#                     Zt12[:, int(len(Yt11) / 2)],
#                     "-",
#                     color="k",
#                     label="L1S2",
#                 )

#                 plt.plot(
#                     Yt21[:, int(len(Yt21) / 2)],
#                     Zt21[:, int(len(Yt21) / 2)],
#                     "-",
#                     color="k",
#                     label="L2S1",
#                 )
#                 plt.plot(
#                     Yt22[:, int(len(Yt22) / 2)],
#                     Zt22[:, int(len(Yt22) / 2)],
#                     "-",
#                     color="k",
#                     label="L2S2",
#                 )

#                 plt.plot(
#                     Ytflat[:, int(len(Ytflat) / 2)],
#                     Ztflat[:, int(len(Ytflat) / 2)],
#                     "-",
#                     color="k",
#                     label="Flat",
#                 )

#                 plt.plot(
#                     Yt31[:, int(len(Yt31) / 2)],
#                     Zt31[:, int(len(Yt31) / 2)],
#                     "-",
#                     color="k",
#                     label="L3S1",
#                 )
#                 plt.plot(
#                     Yt32[:, int(len(Yt32) / 2)],
#                     Zt32[:, int(len(Yt32) / 2)],
#                     "-",
#                     color="k",
#                     label="L3S2",
#                 )
#           #      plot_surface(Xt11, Yt11, Zt11, fig, 15, 85, 'black', half_index=50)
#                 plot_surface(Xt12, Yt12, Zt12, fig, 15, 85, 'black')

#                 plot_surface(Xt21, Yt21, Zt21, fig, 37, 62, 'black')
#                 plot_surface(Xt22, Yt22, Zt22, fig, 37, 62, 'black')

#                 plot_surface(Xtflat, Ytflat, Ztflat, fig, 0, 100, 'black', half_index=100)

#                 plot_surface(Xt31, Yt31, Zt31, fig, 20, 80, 'black', half_index=100)
#                 plot_surface(Xt32, Yt32, Zt32, fig, 20, 80, 'black', half_index=100)

        x_points = [x_rx, x_32, x_31, x_flat,
                    x_22, x_21, x_12, x_11, pos_ap[0]]
        y_points = [y_rx, y_32, y_31, y_flat,
                    y_22, y_21, y_12, y_11, pos_ap[1]]
        z_points = [z_rx, z_32, z_31, z_flat,
                    z_22, z_21, z_12, z_11, pos_ap[2]]

        # These rays aren't going to hit the FTS and are really incorrect here
        # (surfaces defined don't really have boundaries so weird things can happen), so we discard them
        if np.abs(pos_ap[0]) > 2 or np.abs(pos_ap[2]) > 2:
            continue
        # print('plotting...')
        if (plot):
            for i in range(len(x_points) - 1):
                plot_ray([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]],
                         [z_points[i], z_points[i + 1]], fig, col, alph)

            #plt.plot([y_rx, y_32], [z_rx, z_32], "-", color=col, alpha=alph)
            #plt.plot([y_32, y_31], [z_32, z_31], "-", color=col, alpha=alph)
            #plt.plot([y_31, y_flat], [z_31, z_flat], "-", color=col, alpha=alph)
            #plt.plot([y_12, y_11], [z_12, z_11], "-", color=col, alpha=alph)
            #plt.plot([y_21, y_12], [z_21, z_12], "-", color=col, alpha=alph)
            #plt.plot([y_22, y_21], [z_22, z_21], "-", color=col, alpha=alph)
            #plt.plot([y_flat, y_22], [z_flat, z_22], "-", color=col, alpha=alph)
            #plt.plot([y_11, pos_ap[1]], [z_11, pos_ap[2]], "-", color=col, alpha=alph)

        # Write out
        #print('writing out value %s' %ii)
        #out[0, ii] = pos_ap[0]
        #out[1, ii] = pos_ap[1]
        #out[2, ii] = pos_ap[2]
        out[0].append(pos_ap[0])
        out[1].append(pos_ap[1])
        out[2].append(pos_ap[2])

        # out[3, ii] = total_path_length # phase
        out[3].append(total_path_length)
        out[4].append(0)
#         out[4, ii] = np.exp(
#             (-0.5)
#             * ((de_ve) ** 2 + (de_ho) ** 2)
#             / ((24 * np.pi / 180) / (np.sqrt(8 * np.log(2)))) ** 2 #amplitude
       # ) # Setting FWHP = 24 degrees, you should change this

#         out[5, ii] = N_hat_t[0]
#         out[6, ii] = N_hat_t[1]
#         out[7, ii] = N_hat_t[2]

        out[5].append(N_hat_t[0])
        out[6].append(N_hat_t[1])
        out[7].append(N_hat_t[2])

        #out[8, ii] = tan_og_t[0]
        #out[9, ii] = tan_og_t[1]
        #out[10, ii] = tan_og_t[2]

        out[8].append(tan_og_t[0])
        out[9].append(tan_og_t[1])
        out[10].append(tan_og_t[2])

    if (plot):
        plot_surface(Xt11, Yt11, Zt11, fig, 15, 85, 'teal', half_index=50)
        plot_surface(Xt12, Yt12, Zt12, fig, 15, 85, 'teal')

        plot_surface(Xt21, Yt21, Zt21, fig, 37, 62, 'teal')
        plot_surface(Xt22, Yt22, Zt22, fig, 37, 62, 'teal')

        plot_surface(Xtflat, Ytflat, Ztflat, fig,
                     0, 100, 'gray', half_index=100)

        plot_surface(Xt31, Yt31, Zt31, fig, 20, 80, 'teal', half_index=100)
        plot_surface(Xt32, Yt32, Zt32, fig, 20, 80, 'teal', half_index=100)

    return np.array(out)


def get_final_rays(shift, n_linear, theta_bound=.3, y_ap=-.426):
    assert y_ap <= (-.426)
    start_position = [0, ((210+82)*mm_to_in + 1.848), -20.9]
    new_start = np.add(start_position, shift)

    out = aperature_fields(new_start, fts_geo, None, None, n_linear=n_linear,
                           theta_bound=theta_bound, plot=False, y_ap=y_ap)
    # print(out)

    start_rays_mcmahon = [convert_to_ray_mcmahon(out_arr) for out_arr in out.T]
    return start_rays_mcmahon
