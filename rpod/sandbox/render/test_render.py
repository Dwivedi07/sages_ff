import numpy as np
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image

root_folder = Path(__file__).resolve().parent.parent.parent.parent

from rpod.dynamics.dynamics_rot import q2dcm, q_inv
import tobypy

def to_xyz(q):
    """Convert quaternion [w, x, y, z] to 3D vector, flipping sign if w<0."""
    w, x, y, z = q.tolist()
    if w < 0:
        return np.array([-x, -y, -z])
    return np.array([x, y, z])

def dir(v):
    return v / np.linalg.norm(v)


# ----------------- Shirt sequence class ----------------- #

class Shirt:
    """
    Sequence wrapper for shirt-like data.
    
    You can construct it either:
      - directly from arrays (intended for custom trajectories), or
      - from the legacy JSON dataset via Shirt.from_dataset(...).
    """

    def __init__(
        self,
        r_Vo2To_vbs_true: np.ndarray,
        q_vbs2tango_true: np.ndarray,
        rv_eci2com_eci: np.ndarray,
        q_eci2pri: np.ndarray,
        start_time_microsec: int,
        dt_sec: float,
        start_idx: int = 0,
        max_images: int | None = None,
    ):
        """
        Parameters
        ----------
        r_Vo2To_vbs_true : (N, 3) array
            Relative position of target w.r.t. camera (vbs frame).
        q_vbs2tango_true : (N, 4) array
            Quaternion (w,x,y,z) from vbs frame to target (tango) frame.
        rv_eci2com_eci : (N, 6) or (N, 3) array
            Absolute deputy state in ECI; only the position part is used
            (first 3 entries).
        q_eci2pri : (N, 4) array
            Quaternion (w,x,y,z) from ECI frame to primary (body) frame.
        start_time_microsec : int
            GPS time at index 0 in microseconds (consistent with tobypy.sun_position_eci).
        dt_sec : float
            Time step between frames [s].
        start_idx : int
            Starting index in the sequence.
        max_images : int or None
            Maximum number of frames to render. If None, use full length.
        """

        # Store arrays as float32/float64 consistently
        self.r_Vo2To_vbs_true = np.asarray(r_Vo2To_vbs_true)
        self.q_vbs2tango_true = np.asarray(q_vbs2tango_true)
        self.rv_eci2com_eci = np.asarray(rv_eci2com_eci)
        self.q_eci2pri = np.asarray(q_eci2pri)

        # Basic shape check (optional but helpful)
        N = self.r_Vo2To_vbs_true.shape[0]
        assert self.q_vbs2tango_true.shape[0] == N
        assert self.rv_eci2com_eci.shape[0] == N
        assert self.q_eci2pri.shape[0] == N

        self.start_time = int(start_time_microsec)
        self.dt = float(dt_sec)

        self.image_idx = int(start_idx)
        if max_images is None:
            self.image_count = N
        else:
            self.image_count = min(N, start_idx + int(max_images))

    # -------- dataset-based constructor (backwards compatibility) -------- #

    @classmethod
    def from_dataset(
        cls,
        path_to_shirt: Path,
        roe: str,
        start_idx: int = 0,
        max_images: int | None = None,
    ):
        """
        Legacy constructor that replicates your original JSON-based loading.
        """
        with open(path_to_shirt / roe / f"{roe}.json", "r") as f:
            roe_data = json.load(f)

        with open(path_to_shirt / roe / "metadata.json", "r") as f:
            metadata = json.load(f)

        r_Vo2To_vbs_true = np.array(
            [entry["r_Vo2To_vbs_true"] for entry in roe_data]
        )
        q_vbs2tango_true = np.array(
            [entry["q_vbs2tango_true"] for entry in roe_data]
        )

        rv_eci2com_eci = np.array(metadata["sAbsState"]["rv_eci2com_eci"])
        q_eci2pri = np.array(metadata["sAbsState"]["q_eci2pri"])

        # Original start time in GPS microseconds
        start_time = tobypy.date_gps(2011, 7, 18, 1, 0, 0)
        dt = float(metadata["pSim"]["cam_step"])

        return cls(
            r_Vo2To_vbs_true=r_Vo2To_vbs_true,
            q_vbs2tango_true=q_vbs2tango_true,
            rv_eci2com_eci=rv_eci2com_eci,
            q_eci2pri=q_eci2pri,
            start_time_microsec=start_time,
            dt_sec=dt,
            start_idx=start_idx,
            max_images=max_images,
        )

    # ----------------- iteration API ----------------- #

    def done(self):
        return self.image_idx >= self.image_count

    def next(self):
        if self.done():
            return None

        # simple struct-like object
        sample = type("Sample", (), {})()
        sample.image = None  # placeholder if you later attach raw images

        # Relative pose information (target w.r.t. vbs)
        sample.vbs2tango = self.q_vbs2tango_true[self.image_idx]
        sample.Vo2To_vbs = self.r_Vo2To_vbs_true[self.image_idx]

        # ---- sun direction in camera frame ---- #
        dt_microseconds = int(self.dt * 1e6)
        current_time_microseconds = self.start_time + self.image_idx * dt_microseconds

        sun_eci = tobypy.sun_position_eci(current_time_microseconds)
        q_eci2spri = self.q_eci2pri[self.image_idx]

        # only position is needed from absolute state
        r_eci2spri_eci = self.rv_eci2com_eci[self.image_idx][0:3]

        # rotation from ECI to camera frame (via primary)
        R_eci2cam = q2dcm(q_inv(q_eci2spri))
        sample.r_sun_cam = R_eci2cam @ (sun_eci - r_eci2spri_eci)

        self.image_idx += 1
        return sample


# ----------------- rendering helpers ----------------- #

def render_sequence(
    shirt: Shirt,
    noise_offset: int = 0,
    pause_sec: float = 0.1,
    sleep_sec: float = 1.0,
    save_dir: Path | None = None,
):
    """
    Generic renderer for any Shirt sequence (dataset-based or custom data).

    Parameters
    ----------
    shirt : Shirt
        Sequence object providing `.next()`.
    noise_offset : int
        Offset added to the noise index.
    pause_sec : float
        Matplotlib pause between frames.
    sleep_sec : float
        Additional sleep (if you want slower playback).
    """
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

    image_count = noise_offset
    print("Starting rendering sequence")

    frame_idx = 0  # internal frame counter (for saving)
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)   # <-- this ensures it exists
        

    while not shirt.done():
        sample = shirt.next()
        if sample is None:
            break

        # Renderer configuration
        cfg = tobypy.RenderConfig()
        cfg.camera = tobypy.Camera.PointGrey
        cfg.draw_target = tobypy.TargetDrawMethod.DrawSemiResolved
        cfg.r_target = np.array(sample.Vo2To_vbs, dtype=np.float32)
        cfg.q_target = np.array(to_xyz(sample.vbs2tango), dtype=np.float32)
        cfg.dir_sun_cam = np.array(dir(sample.r_sun_cam), dtype=np.float32)
        cfg.draw_stars = True
        cfg.draw_mask = False
        cfg.noise_index = image_count

        # Render image
        image_data = renderer.render(cfg)

        # Render mask
        cfg.draw_mask = True
        mask_data = renderer.render(cfg)

        image_count += 1

        # Plot
        ax1.clear()
        ax1.imshow(image_data, cmap="gray")
        ax1.set_title(f"Image {image_count}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.clear()
        ax2.imshow(mask_data, cmap="gray")
        ax2.set_title(f"Mask {image_count}")
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.draw()
        plt.pause(pause_sec)
        time.sleep(sleep_sec)
        
        if save_dir is not None:
            image_file = save_dir / f"image_{frame_idx:05d}.png"
            mask_file = save_dir / f"mask_{frame_idx:05d}.png"

            plt.imsave(image_file, image_data, cmap="gray")
            plt.imsave(mask_file, mask_data, cmap="gray")

        frame_idx += 1
        
    pngs_to_gif(save_dir, pattern="image_*.png", out_path=save_dir / "sequence.gif", fps=4.0)

    print("Done")
    plt.show()  # keep window open


def pngs_to_gif(
    folder: str | Path,
    pattern: str = "image_*.png",
    out_path: str | Path = "sequence.gif",
    fps: float = 10.0,
):
    """
    Concatenate PNGs into an animated GIF.

    Parameters
    ----------
    folder : str | Path
        Directory containing PNG files.
    pattern : str
        Glob pattern for frame files (e.g., 'image_*.png').
    out_path : str | Path
        Output GIF path.
    fps : float
        Frames per second for the GIF.
    """
    folder = Path(folder)
    out_path = Path(out_path)

    frames = sorted(folder.glob(pattern))
    if not frames:
        raise ValueError(f"No files matching {pattern} in {folder}")

    images = [Image.open(f).convert("P") for f in frames]  # or "RGB"
    duration_ms = int(1000 / fps)

    # Save GIF
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF to {out_path}")


# ----------------- public entry points ----------------- #

def render_from_dataset(path_to_shirt: Path, roe: str, start_idx=0, max_images=50, save_dir: Path | None = None):
    """Backwards-compatible entry point using the JSON shirt dataset."""
    shirt = Shirt.from_dataset(path_to_shirt, roe, start_idx, max_images)
    render_sequence(shirt, noise_offset=start_idx, save_dir=save_dir)

def render_from_data(
    r_Vo2To_vbs_true: np.ndarray,
    q_vbs2tango_true: np.ndarray,
    rv_eci2com_eci: np.ndarray,
    q_eci2pri: np.ndarray,
    start_time_microsec: int,
    dt_sec: float,
    start_idx: int = 0,
    max_images: int | None = None,
    save_dir: Path | None = None,
):
    """
    New entry point: render from custom trajectories directly.
    """
    shirt = Shirt(
        r_Vo2To_vbs_true=r_Vo2To_vbs_true,
        q_vbs2tango_true=q_vbs2tango_true,
        rv_eci2com_eci=rv_eci2com_eci,
        q_eci2pri=q_eci2pri,
        start_time_microsec=start_time_microsec,
        dt_sec=dt_sec,
        start_idx=start_idx,
        max_images=max_images,
    )
    render_sequence(shirt, noise_offset=start_idx, save_dir=save_dir)

def render_with_traj_panel(
    # ---- camera render inputs (top-right panel) ----
    r_serv2targ_vbs: np.ndarray,     # (N,3)
    q_vbs2targ_pri: np.ndarray,      # (N,4)  (same convention as your working render_sequence)
    rv_eci2serv_eci: np.ndarray,     # (N,6) or (N,3)
    q_eci2pri: np.ndarray,           # (N,4)

    # ---- trajectory inputs (bottom-left panel) ----
    rtn_cvx_ct: np.ndarray,          # (Nct,3) [R,T,N]
    rtn_cvx: np.ndarray,             # (N,3)   [R,T,N]
    q_vbs2targ_rtn: np.ndarray,      # (N,4)   for q2dcm()  (servicer/body triad drawn at servicer position)
    q_targ_pri2targ_rtn: np.ndarray, # (N,4)   for q2dcm()  (target attitude triad at RTN origin)

    # ROE / control input
    roe: np.ndarray,        # (N,6)
    u: np.ndarray,          # (N,3)
    tvec_sec: np.ndarray,   # (N,)

    # ---- time / IO ----
    start_time_microsec: int,
    dt_sec: float,
    start_idx: int,
    max_images: int,
    save_dir="figures/panel",
    fps: float = 5.0,
    stride: int = 1,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    N = min(
        len(r_serv2targ_vbs), len(q_vbs2targ_pri), len(rv_eci2serv_eci), len(q_eci2pri),
        len(rtn_cvx), len(q_vbs2targ_rtn), len(q_targ_pri2targ_rtn),
        len(roe), len(u), len(tvec_sec),
    )
    if not (0 <= start_idx < N):
        raise ValueError(f"start_idx={start_idx} out of range [0, {N-1}]")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_images <= 0:
        raise ValueError("max_images must be > 0")

    shirt = Shirt(
        r_Vo2To_vbs_true=r_serv2targ_vbs[:N],
        q_vbs2tango_true=q_vbs2targ_pri[:N],
        rv_eci2com_eci=rv_eci2serv_eci[:N],
        q_eci2pri=q_eci2pri[:N],
        start_time_microsec=start_time_microsec,
        dt_sec=dt_sec,
        start_idx=start_idx,
        max_images=min(N - start_idx, max_images * stride),
    )

    renderer = tobypy.make_renderer()

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    outer = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.5, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.05, hspace=0.05,
    )

    # Left-top: 3D trajectory animation (unchanged content below)
    axL = fig.add_subplot(outer[0, 0], projection="3d")

    # Left-bottom: egocentric rendered animation (was right panel before)
    axR = fig.add_subplot(outer[1, 0])

    # Right-top: 3 stacked stem plots for u
    gs_u = outer[0, 1].subgridspec(3, 1, hspace=0.35)
    ax_u0 = fig.add_subplot(gs_u[0, 0])
    ax_u1 = fig.add_subplot(gs_u[1, 0], sharex=ax_u0)
    ax_u2 = fig.add_subplot(gs_u[2, 0], sharex=ax_u0)

    # Right-bottom: 3 columns for ROE pair plots
    gs_roe = outer[1, 1].subgridspec(1, 3, wspace=0.35)
    ax_r0 = fig.add_subplot(gs_roe[0, 0])
    ax_r1 = fig.add_subplot(gs_roe[0, 1])
    ax_r2 = fig.add_subplot(gs_roe[0, 2])

    # ----------------- STATIC RIGHT-SIDE PLOTS (NEW) -----------------
    # u stems
    for ax, i, lab in [(ax_u0, 0, "$\Delta v_R"), (ax_u1, 1, "$\Delta v_T"), (ax_u2, 2, "$\Delta v_N")]:
        ax.stem(tvec_sec[:N], u[:N, i], basefmt=" ",)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    ax_u0.set_title("Control inputs")
    ax_u2.set_xlabel("t [s]")
    plt.setp(ax_u0.get_xticklabels(), visible=False)
    plt.setp(ax_u1.get_xticklabels(), visible=False)

    # ROE pair plots
    ax_r0.plot(roe[:, 1], roe[:, 0])
    ax_r1.plot(roe[:, 2], roe[:, 3])
    ax_r2.plot(roe[:, 4], roe[:, 5])
    ax_r0.scatter(roe[0, 1], roe[0, 0], marker="o", color="r", edgecolor="k", zorder=20)
    ax_r0.scatter(roe[-1, 1], roe[-1, 0], marker="o", color="g", edgecolor="k", zorder=20)
    ax_r1.scatter(roe[0, 2], roe[0, 3], marker="o", color="r", edgecolor="k", zorder=20)
    ax_r1.scatter(roe[-1, 2], roe[-1, 3], marker="o", color="g", edgecolor="k", zorder=20)
    ax_r2.scatter(roe[0, 4], roe[0, 5], marker="o", color="r", edgecolor="k", zorder=20)
    ax_r2.scatter(roe[-1, 4], roe[-1, 5], marker="o", color="g", edgecolor="k", zorder=20)
    # ax_r0.axis('box')
    # ax_r1.axis('box')
    # ax_r2.axis('box')
    ax_r0.set_xlabel(r"$a\delta \lambda$ [m]")
    ax_r0.set_ylabel(r"$a\delta a$ [m]")
    ax_r1.set_xlabel(r"$a\delta e_x$ [m]")
    ax_r1.set_ylabel(r"$a\delta e_y$ [m]")
    ax_r2.set_xlabel(r"$a\delta i_x$ [m]")
    ax_r2.set_ylabel(r"$a\delta i_y$ [m]")
    for ax in (ax_r0, ax_r1, ax_r2):
        ax.grid(True, alpha=0.3)

    # ----------------- existing left-side setup continues below -----------------

    # ---- left static ----
    axL.set_facecolor("white")
    axL.grid(False)
    for axis in (axL.xaxis, axL.yaxis, axL.zaxis):
        axis.pane.fill = False
        axis.line.set_color("black")
        axis.line.set_linewidth(1)

    axL.plot(rtn_cvx_ct[:, 1], rtn_cvx_ct[:, 2], rtn_cvx_ct[:, 0],
             color="C1", linewidth=2, zorder=2, label="Nominal (CT)")

    axL.scatter(rtn_cvx_ct[0, 1],  rtn_cvx_ct[0, 2],  rtn_cvx_ct[0, 0],
                marker="o", color="r", edgecolor="k", zorder=20, label="Start")
    axL.scatter(rtn_cvx_ct[-1, 1], rtn_cvx_ct[-1, 2], rtn_cvx_ct[-1, 0],
                marker="o", color="g", edgecolor="k", zorder=20, label="Goal")

    k0 = start_idx
    R0, T0, N0 = rtn_cvx[k0, 0], rtn_cvx[k0, 1], rtn_cvx[k0, 2]
    ptL, = axL.plot([T0], [N0], [R0], marker="o", markersize=5)

    axL.set_xlabel("$T$, m")
    axL.set_ylabel("$N$, m")
    axL.set_zlabel("$R$, m")
    axL.view_init(elev=25, azim=70)

    # add triad color legend (proxy handles) under existing handles
    triad_handles = [
        Line2D([0], [0], color="r", lw=3),
        Line2D([0], [0], color="g", lw=3),
        Line2D([0], [0], color="b", lw=3),
    ]
    triad_labels = ["x-axis", "y-axis", "z-axis"]
    handles, labels = axL.get_legend_handles_labels()
    handles += triad_handles
    labels  += triad_labels
    axL.legend(handles, labels, loc="upper left")

    mins = np.min(rtn_cvx_ct, axis=0)   # [R,T,N]
    maxs = np.max(rtn_cvx_ct, axis=0)
    ctr  = 0.5 * (mins + maxs)
    rad  = 0.5 * np.max(maxs - mins)
    rad = float(rad) if rad > 0 else 1.0
    axL.set_xlim(ctr[1] - rad, ctr[1] + rad)
    axL.set_ylim(ctr[2] - rad, ctr[2] + rad)
    axL.set_zlim(ctr[0] - rad, ctr[0] + rad)

    axis_len = 0.1 * rad
    quivs_serv = []
    quivs_targ = []

    # ---- right (image) init ----
    axR.set_title("Egocentric render")
    axR.set_xticks([]); axR.set_yticks([])
    im = axR.imshow(np.zeros((480, 640), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
    time_text = axR.text(
        0.02, 0.98, "", transform=axR.transAxes,
        ha="left", va="top", fontsize=11, color="white",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=3.0)
    )
    axR.set_aspect("equal", adjustable="box")
    
    fig.tight_layout()

    # ----------------- animation part below: DO NOT MODIFY -----------------
    frame_out = 0
    while (not shirt.done()) and (frame_out < max_images):
        sample = None
        for _ in range(stride):
            sample = shirt.next()
            if sample is None:
                break
        if sample is None:
            break

        k = shirt.image_idx - 1
        if k >= N:
            break

        # ---- right render (same logic as render_sequence) ----
        cfg = tobypy.RenderConfig()
        cfg.camera = tobypy.Camera.PointGrey
        cfg.draw_target = tobypy.TargetDrawMethod.DrawSemiResolved
        cfg.r_target = np.asarray(sample.Vo2To_vbs, dtype=np.float32)
        cfg.q_target = np.asarray(to_xyz(sample.vbs2tango), dtype=np.float32)
        cfg.dir_sun_cam = np.asarray(dir(sample.r_sun_cam), dtype=np.float32)
        cfg.draw_stars = True
        cfg.draw_mask = False
        cfg.noise_index = int(frame_out)

        image_data = renderer.render(cfg)
        im.set_data(image_data)

        t_sec = float((k - start_idx) * dt_sec)
        time_text.set_text(f"t = {t_sec:.1f} s")

        # ---- left update: servicer point ----
        Rk, Tk, Nk = rtn_cvx[k, 0], rtn_cvx[k, 1], rtn_cvx[k, 2]
        ptL.set_data([Tk], [Nk])
        ptL.set_3d_properties([Rk])

        # remove previous triads
        for q in quivs_serv:
            try: q.remove()
            except Exception: pass
        quivs_serv = []

        for q in quivs_targ:
            try: q.remove()
            except Exception: pass
        quivs_targ = []

        # ---- servicer/body triad (at servicer position) ----
        Rmat = q2dcm(q_vbs2targ_rtn[k])  # columns are body axes in [R,T,N]
        ex, ey, ez = Rmat[:, 0], Rmat[:, 1], Rmat[:, 2]
        quivs_serv.append(axL.quiver(Tk, Nk, Rk, axis_len*ex[1], axis_len*ex[2], axis_len*ex[0], color="r", zorder=10))
        quivs_serv.append(axL.quiver(Tk, Nk, Rk, axis_len*ey[1], axis_len*ey[2], axis_len*ey[0], color="g", zorder=10))
        quivs_serv.append(axL.quiver(Tk, Nk, Rk, axis_len*ez[1], axis_len*ez[2], axis_len*ez[0], color="b", zorder=10))

        # ---- target triad (at RTN origin) ----
        Rmat_t = q2dcm(q_targ_pri2targ_rtn[k])  # columns are target axes in [R,T,N]
        ex_t, ey_t, ez_t = Rmat_t[:, 0], Rmat_t[:, 1], Rmat_t[:, 2]
        Tt, Nt, Rt = 0.0, 0.0, 0.0
        quivs_targ.append(axL.quiver(Tt, Nt, Rt, axis_len*ex_t[1], axis_len*ex_t[2], axis_len*ex_t[0], color="r", zorder=10))
        quivs_targ.append(axL.quiver(Tt, Nt, Rt, axis_len*ey_t[1], axis_len*ey_t[2], axis_len*ey_t[0], color="g", zorder=10))
        quivs_targ.append(axL.quiver(Tt, Nt, Rt, axis_len*ez_t[1], axis_len*ez_t[2], axis_len*ez_t[0], color="b", zorder=10))

        fig.savefig(save_dir / f"frame_{frame_out:05d}.png", dpi=150)
        frame_out += 1

    pngs_to_gif(save_dir, pattern="frame_*.png", out_path=save_dir / "panel.gif", fps=fps)
    plt.close(fig)




if __name__ == "__main__":
    
    save_dir = Path("/home/yuji/github/art_vla/rpod/sandbox/render_output")    

    # --- old behavior (dataset-based) ---
    # path_to_shirt = Path("/home/yuji/dataset/shirtv1/")
    # roe = "roe2"
    # render_from_dataset(path_to_shirt, roe, start_idx=0, max_images=10, save_dir=save_dir)

    # --- example for custom data (pseudo-code; replace with real arrays) ---
    
    servicer_data_path = Path("/home/yuji/dataset/shirtv1/roe2/metadata.json")
    target_data_path = Path("/home/yuji/dataset/shirtv1/roe2/roe2.json")
    
    with open(servicer_data_path, 'r') as f:
        servicer_data = json.load(f)
                
    with open(target_data_path, 'r') as f:
        target_data = json.load(f)
    
    r_Vo2To_vbs_true = np.array([entry['r_Vo2To_vbs_true'] for entry in target_data])
    q_vbs2tango_true = np.array([entry['q_vbs2tango_true'] for entry in target_data])
    
    servicer_metadata = servicer_data["sAbsState"]
    rv_eci2com_eci = np.array(servicer_metadata["rv_eci2com_eci"])
    q_eci2pri = np.array(servicer_metadata["q_eci2pri"])
    
    dt_sec = servicer_data["pSim"]["cam_step"]
    start_time = tobypy.date_gps(2028, 1, 1, 0, 0, 0)  # or your own start time
    render_from_data(
        r_Vo2To_vbs_true,
        q_vbs2tango_true,
        rv_eci2com_eci,
        q_eci2pri,
        start_time_microsec=start_time,
        dt_sec=dt_sec,
        start_idx=0,
        max_images=10,
        save_dir=save_dir
    )
