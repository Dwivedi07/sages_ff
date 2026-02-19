import numpy as np
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt

import tobypy

def to_xyz(q):
    w, x, y, z = q.tolist()
    if w < 0:
        return np.array([-x, -y, -z])
    return np.array([x, y, z])

def quat_inv(q):
    w, x, y, z = q.tolist()
    return np.array([w, -x, -y, -z])

def dcm_from_quat(q: np.array):
    a, b, c, d = q.tolist()
    aa = a * a
    ab = a * b
    ac = a * c
    ad = a * d
    bb = b * b
    bc = b * c
    bd = b * d
    cc = c * c
    cd = c * d
    dd = d * d
    ax = aa + bb - cc - dd
    ay = 2. * (bc - ad)
    az = 2. * (bd + ac)
    bx = 2. * (bc + ad)
    by = aa - bb + cc - dd
    bz = 2. * (cd - ab)
    cx = 2. * (bd - ac)
    cy = 2. * (cd + ab)
    cz = aa - bb - cc + dd
    return np.array([ax, ay, az, bx, by, bz, cx, cy, cz]).reshape(3, 3)

def dir(v):
    return v / np.linalg.norm(v)

class Shirt():
    def __init__(self, path_to_shirt, roe, start_idx=0, max_images=None):
        """
            path_to_shirt [Path]: absolute path to shirt dataset
            roe [String]: roe1 or roe2
        """
        self.path_to_shirt = path_to_shirt
        self.roe = roe 
        self.image_idx = start_idx
        self.start_time = tobypy.date_gps(2011, 7, 18, 1, 0, 0)

        with open(path_to_shirt / roe / f'{roe}.json', 'r') as f:
            roe_data = json.load(f)
        
        with open(path_to_shirt / roe / f'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Extract the required data (vbs: vision-based sensor (i.e., camera) frame) 
        # Vo: Vision origin (servicer), To: Tango origin (target)
        self.r_Vo2To_vbs_true = np.array([entry['r_Vo2To_vbs_true'] for entry in roe_data])
        self.q_vbs2tango_true = np.array([entry['q_vbs2tango_true'] for entry in roe_data])
        
        # pri: principle axes (body frame) 
        self.rv_eci2com_eci = np.array(metadata['sAbsState']['rv_eci2com_eci'])
        self.q_eci2pri = np.array(metadata['sAbsState']['q_eci2pri'])

        if max_images is None:
            self.image_count = len(roe_data)
        else:
            self.image_count = max_images
        self.dt = float(metadata['pSim']['cam_step'])
    
    def done(self):
        return self.image_idx >= self.image_count

    def next(self):
        if self.done():
            return None
        
        sample = type('Sample', (), {})()
        sample.image = None

        # Relative pose information
        sample.vbs2tango = self.q_vbs2tango_true[self.image_idx]
        sample.Vo2To_vbs = self.r_Vo2To_vbs_true[self.image_idx]

        # Calculate sun direction
        dt_microseconds = int(self.dt * 1e6)
        current_time_microseconds = self.start_time + self.image_idx * dt_microseconds
        sun_eci = tobypy.sun_position_eci(current_time_microseconds)
        q_eci2spri = self.q_eci2pri[self.image_idx]
        r_eci2spri_eci = self.rv_eci2com_eci[self.image_idx][0:3]
        R_eci2cam = dcm_from_quat(quat_inv(q_eci2spri))
        sample.r_sun_cam = R_eci2cam @ (sun_eci - r_eci2spri_eci)

        self.image_idx += 1

        return sample
        
def main(path_to_shirt, roe, start_idx=0, max_images=10):
    renderer = tobypy.make_renderer()
    
    # Matplotlib viewer with side-by-side display
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title("Rendered Image")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title("Mask")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()

    # Load shirt
    shirt = Shirt(path_to_shirt, roe, start_idx, max_images)
    image_count = start_idx
    print("Loaded shirt")

    while not shirt.done():
        sample = shirt.next()

        # Create renderer config
        cfg = tobypy.RenderConfig()
        cfg.camera = tobypy.Camera.PointGrey
        cfg.draw_target = tobypy.TargetDrawMethod.DrawSemiResolved
        cfg.r_target = np.array(sample.Vo2To_vbs, dtype=np.float32)
        cfg.q_target = np.array(to_xyz(sample.vbs2tango), dtype=np.float32)
        cfg.dir_sun_cam = np.array(dir(sample.r_sun_cam), dtype=np.float32)
        cfg.draw_stars = True
        cfg.draw_mask = False
        cfg.noise_index = image_count

        # Render the regular scene
        image_data = renderer.render(cfg)

        # Render the mask
        cfg.draw_mask = True
        mask_data = renderer.render(cfg)

        image_count += 1

        # Show images side by side
        ax1.clear()
        ax1.imshow(image_data, cmap='gray')
        ax1.set_title(f"Image {image_count}")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        ax2.clear()
        ax2.imshow(mask_data, cmap='gray')
        ax2.set_title(f"Mask {image_count}")
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.draw()
        plt.pause(0.1)

        time.sleep(1.0)

    print("Done")
    plt.show()  # Keep window open

if __name__ == "__main__":
    path_to_shirt = Path("/home/yuji/dataset/shirtv1/")
    roe = "roe2"
    start_idx = 0
    max_images = 50
    main(path_to_shirt, roe, start_idx, max_images)
