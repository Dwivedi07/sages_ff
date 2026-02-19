################ Low Earth Orbit RPO Design Reference Mission Parameters
import sys
import numpy as np
import scipy.stats as stats

from pathlib import Path
root_folder = Path(__file__).resolve().parent.parent.parent  # /art_lang/
sys.path.append(str(root_folder))

from rpod.optimization.scvx import SCVxParams 
from rpod.dynamics.dynamics_trans import mu_E, propagate_oe

##########################################################################################
################################### PARAMETERS ###########################################
##########################################################################################

# Problem dimensions #####################################################################
N_STATE = 6
N_ACTION = 3

########## HELPER FUNCTIONS ##############

def generate_koz(dim_arr, n_time, t_switch=None):
    """
    Generate time-variant DEED matrix and ellipsoid surfaces for plotting.
    
    Parameters:
    -----------
    dim_arr : array_like
        - If 1D (3 elements): Single ellipsoid dimensions [x, y, z]
        - If 2D (N x 3): N ellipsoid dimensions, each row is [x, y, z]
    n_time : int
        Number of time steps
    t_switch : array_like, optional
        Time indices when to switch ellipsoids (length N-1).
        If None or empty, uses single ellipsoid for all time.
    
    Returns:
    --------
    DEED : ndarray, shape (n_time, 6, 6)
        Time-variant ellipsoid constraint matrices
    x_ell, y_ell, z_ell : ndarray, shape (N, 100, 100)
        Ellipsoid surfaces for plotting (N ellipsoids)
    """
    
    # Position selection matrix (only position, not velocity)
    D_pos = np.eye(3, 6, dtype=float)
    
    # Handle input dimensions
    dim_arr = np.atleast_2d(dim_arr)
    if dim_arr.shape[1] != 3:
        dim_arr = dim_arr.T  # Transpose if needed
    
    n_ellipsoids = dim_arr.shape[0]
    
    # Handle time switching
    if t_switch is None or len(t_switch) == 0:
        # Single ellipsoid case
        t_switch = [n_time]
    else:
        t_switch = list(t_switch) + [n_time]  # Add final time
    
    # Validate dimensions
    if len(t_switch) != n_ellipsoids:
        raise ValueError(f"Number of switch times ({len(t_switch)}) must match number of ellipsoids ({n_ellipsoids})")
    
    # Pre-compute DEED matrices for each ellipsoid
    DEED_list = []
    for i in range(n_ellipsoids):
        dim = dim_arr[i]
        E = np.diag([1/dim[0], 1/dim[1], 1/dim[2]])
        ED = D_pos * np.diag(E)[:, np.newaxis]
        DEED_i = ED.T @ ED
        DEED_list.append(DEED_i)
    
    # Build time-variant DEED array
    DEED = np.empty((n_time, 6, 6), dtype=float)
    t_start = 0
    
    for i, t_end in enumerate(t_switch):
        t_end = min(t_end, n_time)  # Ensure we don't exceed n_time
        if t_start < t_end:
            DEED[t_start:t_end] = np.tile(
                DEED_list[i][np.newaxis, :, :], 
                (t_end - t_start, 1, 1)
            )
        t_start = t_end
    
    # Generate ellipsoid surfaces for plotting
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    x_ell = np.zeros((n_ellipsoids, 100, 100))
    y_ell = np.zeros((n_ellipsoids, 100, 100))
    z_ell = np.zeros((n_ellipsoids, 100, 100))
    
    for i in range(n_ellipsoids):
        dim = dim_arr[i]
        rx, ry, rz = dim[0], dim[1], dim[2]
        
        x_ell[i] = rx * np.outer(np.cos(u), np.sin(v))
        y_ell[i] = ry * np.outer(np.sin(u), np.sin(v))
        z_ell[i] = rz * np.outer(np.ones_like(u), np.cos(v))
        
    r_ell = np.stack([x_ell, y_ell, z_ell], axis=1)

    return DEED, r_ell

def sample_reset_condition(rng=None, behavior=None):
    """
    Sample initial/final conditions and waypoints with reproducible randomness.
    Pass in a numpy.random.Generator (rng) to avoid global RNG state.
    """

    if rng is None:
        rng = np.random.default_rng()

    # initial condition
    roe_0 = np.array([0, -120, 0, 5, 0, 5], dtype=float)
    roe_0[1] += rng.integers(-10, 10) / 10 * 20
    roe_0[2] += rng.integers(-10, 10) / 10 * 4
    roe_0[3] += rng.integers(-10, 10) / 10 * 4
    roe_0[4] += rng.integers(-10, 10) / 10 * 4
    roe_0[5] += rng.integers(-10, 10) / 10 * 4

    if behavior is None:
        behavior = rng.integers(0, 7)

    if behavior == 0:  # get around KOZ
        roe_f = np.array([0, 0, 0, 32, 0, 32], dtype=float)
        roe_f[2] += rng.integers(-10, 10) / 10 * 2
        roe_f[3] += rng.integers(-10, 10) / 10 * 2
        roe_f[4] += rng.integers(-10, 10) / 10 * 2
        roe_f[5] += rng.integers(-10, 10) / 10 * 2
        t_idx_wyp, wyp = [], []

    elif behavior == 1:  # KOZ (fast)
        roe_f = np.array([0, 0, 0, 32, 0, 32], dtype=float)
        roe_f[2] += rng.integers(-10, 10) / 10 * 2
        roe_f[3] += rng.integers(-10, 10) / 10 * 2
        roe_f[4] += rng.integers(-10, 10) / 10 * 2
        roe_f[5] += rng.integers(-10, 10) / 10 * 2
        t_idx_wyp = [int(0.7 * n_time)]
        t_idx_wyp[0] += rng.integers(-5, 5)
        wyp = [roe_f.copy()]

    elif behavior == 2:  # -30 m hold
        roe_f = np.array([0, -35, 0, 0, 0, 0], dtype=float)
        roe_f[1] += rng.integers(-10, 10) / 10 * 5
        roe_f[2] += rng.integers(-10, 10) / 10 * 2
        roe_f[3] += rng.integers(-10, 10) / 10 * 2
        roe_f[4] += rng.integers(-10, 10) / 10 * 2
        roe_f[5] += rng.integers(-10, 10) / 10 * 2
        t_idx_wyp, wyp = [], []

    elif behavior == 3:  # -30 m hold (fast)
        roe_f = np.array([0, -35, 0, 0, 0, 0], dtype=float)
        roe_f[1] += rng.integers(-10, 10) / 10 * 5
        roe_f[2] += rng.integers(-10, 10) / 10 * 2
        roe_f[3] += rng.integers(-10, 10) / 10 * 2
        roe_f[4] += rng.integers(-10, 10) / 10 * 2
        roe_f[5] += rng.integers(-10, 10) / 10 * 2
        t_idx_wyp = [int(0.7 * n_time)]
        t_idx_wyp[0] += rng.integers(-5, 5)
        wyp = [roe_f.copy()]

    elif behavior == 4:  # fuel-optimal flyby
        roe_f = np.array([0, 120, 0, 5, 0, 5], dtype=float)
        roe_f[1] += rng.integers(-10, 10) / 10 * 20
        roe_f[2] += rng.integers(-10, 10) / 10 * 2
        roe_f[3] += rng.integers(-10, 10) / 10 * 2
        roe_f[4] += rng.integers(-10, 10) / 10 * 2
        roe_f[5] += rng.integers(-10, 10) / 10 * 2
        t_idx_wyp, wyp = [], []

    elif behavior == 5:  # E/I-separated flyby
        roe_f = np.array([0, 120, 0, 5, 0, 5], dtype=float)
        roe_f[1] += rng.integers(-10, 10) / 10 * 20
        roe_f[2] += rng.integers(-10, 10) / 10 * 2
        roe_f[3] += rng.integers(-10, 10) / 10 * 2
        roe_f[4] += rng.integers(-10, 10) / 10 * 2
        roe_f[5] += rng.integers(-10, 10) / 10 * 2

        t_idx_wyp = [int(0.1 * n_time), int(0.9 * n_time)]
        t_idx_wyp[0] += rng.integers(-2, 2)
        t_idx_wyp[1] += rng.integers(-2, 2)

        def w(seed):
            return 25 + rng.integers(-10, 10) / 10 * 2

        wyp0 = np.array([0, roe_0[1], 0, w(0), 0, w(1)], dtype=float)
        wyp1 = np.array([0, roe_f[1], 0, w(2), 0, w(3)], dtype=float)
        wyp = [wyp0, wyp1]

    # not including abort because it is trivial
    elif behavior == 6:  # recede (abort)
        roe_mid = np.array([0, 0, 0, 32, 0, 32], dtype=float)
        roe_mid[2] += rng.integers(-10, 10) / 10 * 2
        roe_mid[3] += rng.integers(-10, 10) / 10 * 2
        roe_mid[4] += rng.integers(-10, 10) / 10 * 2
        roe_mid[5] += rng.integers(-10, 10) / 10 * 2
        t_idx_wyp = [int(0.4 * n_time), int(0.7 * n_time)]
        t_idx_wyp[0] += rng.integers(-5, 5)
        t_idx_wyp[1] += rng.integers(-5, 5)
        wyp = [roe_mid, roe_mid]
        roe_f = np.array([0, 120, 0, 10, 0, 10])
        roe_f[1] += rng.integers(-10, 10) / 10 * 20
        roe_f[2] += rng.integers(-10, 10) / 10 * 2
        roe_f[3] += rng.integers(-10, 10) / 10 * 2
        roe_f[4] += rng.integers(-10, 10) / 10 * 2
        roe_f[5] += rng.integers(-10, 10) / 10 * 2

    else:
        raise ValueError("behavior not recognized")

    return behavior, roe_0, roe_f, t_idx_wyp, wyp

def sample_reset_condition2(rng=None, behavior=None, det=False):
    """
    Sample initial/final conditions and waypoints with reproducible randomness.
    Pass in a numpy.random.Generator (rng) to avoid global RNG state.
    """

    if rng is None:
        rng = np.random.default_rng()

    # initial condition
    roe_0 = np.array([0, -120, 0, 5, 0, 5], dtype=float)
    if not det: 
        roe_0[1] += rng.integers(-10, 10) / 10 * 20
        roe_0[2] += rng.integers(-10, 10) / 10 * 4
        roe_0[3] += rng.integers(-10, 10) / 10 * 4
        roe_0[4] += rng.integers(-10, 10) / 10 * 4
        roe_0[5] += rng.integers(-10, 10) / 10 * 4

    if behavior is None:
        behavior = rng.integers(0, 6)

    if behavior == 0:  # approach and circumnavigate KOZ
        roe_f = np.array([0, 0, 0, 32, 0, 32], dtype=float)
        t_idx_wyp = [int(0.8 * n_time)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 5
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-10, 9)
        wyp = [roe_f.copy()]

    elif behavior == 1:  # dock
        roe_f = np.array([0, -35, 0, 0, 0, 0], dtype=float)
        t_idx_wyp = [int(0.8 * n_time)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 5
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-10, 9)
        wyp = [roe_f.copy()]

    elif behavior == 2:  # flyby (under KOZ)
        roe_f = np.array([0, 150, 0, 5, 0, 5], dtype=float)
        t_idx_wyp = [int(0.9 * n_time)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-10, 4)
        wyp = [roe_f.copy()]
        
    elif behavior == 3:  # flyby (E/I-separated)
        roe_f = np.array([0, 120, 0, 5, 0, 5], dtype=float)
        wyp0 = np.array([0, roe_0[1], 0, 25, 0, 25], dtype=float)
        wyp1 = np.array([0, roe_f[1], 0, 25, 0, 25], dtype=float)
        t_idx_wyp = [int(0.2 * n_time), int(0.8 * n_time)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-5, 5)
            t_idx_wyp[1] += rng.integers(-5, 5)
            wyp0 = np.array([0, roe_0[1], 0, 25 + rng.integers(-10, 10) / 10 * 2, 0, 25 + rng.integers(-10, 10) / 10 * 2], dtype=float)
            wyp1 = np.array([0, roe_f[1], 0, 25 + rng.integers(-10, 10) / 10 * 2, 0, 25 + rng.integers(-10, 10) / 10 * 2], dtype=float)
        wyp = [wyp0, wyp1]

    elif behavior == 4:  # approach, circumnavigate, and forward 
        roe_f = np.array([0, 120, 0, 35, 0, 35], dtype=float)
        wyp0 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        t_idx_wyp = [int(0.5 * n_time), int(0.7 * n_time)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-5, 0)
            t_idx_wyp[1] += rng.integers(0, 5)
            wyp0 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
            wyp1 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
        wyp = [wyp0, wyp1]

    elif behavior == 5:  # approach, circumnavigate, and retreat 
        roe_f = np.array([0, -120, 0, 35, 0, 35], dtype=float)
        wyp0 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        t_idx_wyp = [int(0.5 * n_time), int(0.7 * n_time)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-5, 0)
            t_idx_wyp[1] += rng.integers(0, 5)
            wyp0 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
            wyp1 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
        wyp = [wyp0, wyp1]

    else:
        raise ValueError("behavior not recognized")

    return behavior, roe_0, roe_f, t_idx_wyp, wyp

# RPOD scenario specification #############################################################

scpparam = SCVxParams()

# Canonical command map (6 modalities)
COMMAND_LIST = {
    0: "Approach to the relative orbit around the target, and circumnavigate",
    1: "Go to -V-bar waypoint, and hold",
    2: "Fast flyby under KOZ, from -V-bar (anti-velocity direction) to +V-bar (velocity direction)",
    3: "Flyby (slow, using E/I separation), from -V (ant-velocity) to +V-bar (velocity direction)",
    4: "approach to the target from -V-bar (anti-velocity direction), circumnavigate, then move to the +V-bar direction (abort maneuver)",
    5: "approach to the target from -V-bar (anti-velocity direction), circumnavigate, then move back to the -V-bar direction (abort maneuver) with RN-plane separation",
}

# per behavior/mode, list the ONLY placeholders allowed in templates
ALLOWED_PLACEHOLDERS = {
    0: ["T_appr_orbits"],
    1: ["T_appr_orbits", "d_lambda_meters"],
    2: ["T_appr_orbits", "d_lambda_meters"],
    3: ["T_EI_sep_orbits", "T_transfer_orbits"],  # "T_settle_orbits"
    4: ["T_appr_orbits", "T_circ_orbits"],   # "T_evac_orbits"
    5: ["T_appr_orbits", "T_circ_orbits"],   # "T_evac_orbits"
}

# for dummy command version
# COMMAND_LIST = {
#     0: "Abort the mission and escape",
#     1: "grasp the target satellite",
# }
# ALLOWED_PLACEHOLDERS = {
#     0: ["T_appr_orbits"],
#     1: ["T_appr_orbits"],
# }


# shared specification 
n_time = 50
n_time_max = 50
n_safe = 50
state = 'roe'

# time dilation (for test) 
# n_time = 8 
# n_time_max = 8
# n_safe = 20
# state = 'roe'

# time discretization
oec0 = np.array([6738.14e3, 0.0005581, np.deg2rad(51.6418), np.deg2rad(301.0371), np.deg2rad(26.1813), np.deg2rad(68.2333)]) 
n = np.sqrt(mu_E/oec0[0]**3)
period = 2*np.pi/n   # seconds

t0_sec = 0
tf_sec = 5 * period
t_safe_sec = 1 * period  # seconds
tvec_sec = np.linspace(t0_sec, tf_sec, n_time)  # nominal dt
dt_sec = tvec_sec[1] - tvec_sec[0]   # seconds
dt_safe_sec = t_safe_sec / n_safe    # seconds

oec = propagate_oe(oec0, tvec_sec)

# Waypoints [m, m, m, m/s, m/s, m/s]
rtn0 = np.array([-4e3, -17.5e3, 0, 0, 6.849, 0])
rtnf = np.array([0, 750, 0, 0, 0, 0])

# dim_koz = np.array([[1000, 1000, 1000], [500, 500, 500]])  
# t_switch = [int(n_time*0.36)]
dim_koz = np.array([[25, 25, 25]])  
t_switch = []
DEED, r_ell = generate_koz(dim_koz, n_time, t_switch=t_switch) 

# Chance constraining,  invICDF = stats.norm.ppf(1-delta_chance)
invICDF = 3.0

# variable scaling (put a rough order of your variables)
Ds = np.eye(N_STATE) * 1.0e3  # [m] for the state
Da = np.diag([1.0,1.0,1.0])

# Navigation
use_nav_artms = True
# Digital
digital_relative_std = np.array([1e-2, 1e-2, 1e-2, 25e-6, 25e-6, 25e-6]) #[m,m,m,m/s,m/s,m/s]
digital_absolute_std = np.array([10., 10., 10., 0.5, 0.5, 0.5]) #[m,m,m,m/s,m/s,m/s]
S_digital_rel = np.diag(digital_relative_std)
Sigma_nav_digital_rtn = S_digital_rel @ S_digital_rel.T
# ARTMS (Kruger Ph.D. thesis)
artms_scale_range_1e5 = np.array([4e-5, 4e-3, 4e-5, 2e-5, 2e-5, 4e-5])
artms_scale_range_1e3 = np.array([1e-4, 4e-3, 2e-3, 2e-3, 2e-3, 2e-3]) 

# Process noise
Q = np.diag([1e-3,1e-3,1e-3,1e-3,1e-3,1e-3])   # per each dt_safe. assume only actuation error / unmodeled process noise for now (Note : this should be just process noise along the uncontrolled trajectory)
QQ = Q @ Q.T

u_max = 5 # [m/s^2], this is not really effective in the current scenario

# Actuation
use_gates_model = True
# Proportional
actuation_noise_std = [0.05, 0.05, 0.05] # [%] Note : this should be improved see model used by BLUE
# Gates model [simga_s, sigma_p, sigma_r, sigma_a], reference: Berning Jr. et al. 2023
sigma_gates = np.array([2e-3, 0.3e-3, 3e-4, 0.3e-3])  

scp_iter_max = scpparam.iter_max 


