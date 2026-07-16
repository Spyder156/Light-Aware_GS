"""Light-Aware GS panel: full in-GUI workflow.
  (1) Load the DiLiGenT bear dataset -> view cameras/images.
  (2) Recover albedo (LIVE): spawns our multi-view de-light optimizer (fullcircle env); a grey bear appears
      then sharpens into recovered albedo, streamed into the viewer as it trains.
  (3) Capture albedo + move a light -> relight live.
Streaming is driven from draw() (fires reliably while the panel is open) with lf.on_frame as a backup, and
every callback is exception-guarded so a transient error can never remove the panel. Errors -> ~/.lichtfeld/light_aware_debug.log."""

import os
import json
import math
import subprocess
import traceback
import numpy as np
import lichtfeld as lf
from lfs_plugins.types import Panel

DBG = os.path.join(os.path.expanduser("~"), ".lichtfeld", "light_aware_debug.log")
def _dbg(msg):
    try:
        with open(DBG, "a") as f: f.write(msg + "\n")
    except Exception: pass

TITLE = (0.3, 0.7, 1.0, 1.0)
DIM = (0.6, 0.6, 0.6, 1.0)
OK = (0.4, 0.9, 0.5, 1.0)
WARN = (1.0, 0.7, 0.3, 1.0)

_HERE = os.path.dirname(os.path.realpath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
DATASET = os.path.join(_REPO, "data", "lfs_bear")
RECOVER = os.path.join(_REPO, "src", "stage6_lichtfeld_integration", "recover_albedo_live.py")
LIVE = os.path.join(_REPO, "outputs", "rt", "lfs_live")
FC_PY = os.path.expanduser("~/miniconda3/envs/fullcircle/bin/python")
INIT_PLY = os.path.join(LIVE, "bear_init.ply")
ALBEDO_NPY = os.path.join(LIVE, "albedo_live.npy")
PROGRESS = os.path.join(LIVE, "progress.json")
NODE = "bear_init"                                                     # scene node name (from the ply filename)


def _to_tensor(a):
    t = lf.Tensor.from_numpy(np.ascontiguousarray(a.astype(np.float32)))
    try: t = t.cuda()
    except Exception: pass
    return t


def _set_colors(sd, arr):
    """Write colors AND tell the renderer to re-upload -- LF draws from a combined-model cache, so an
    in-place set_colors_rgb is invisible without invalidate_cache()+notify_changed()."""
    try:
        sd.set_colors_rgb(_to_tensor(arr))
    except Exception as e:
        _dbg(f"  set_colors_rgb FAILED: {e}\n{traceback.format_exc()}")
        raise
    scene = lf.get_scene()
    if scene is not None:
        for fn in ("invalidate_cache", "notify_changed"):            # LF renders from a combined cache -> must signal both
            try: getattr(scene, fn)()
            except Exception: pass


def _normals_from_gaussians(rot, scl):
    q = np.asarray(rot, dtype=np.float64); s = np.asarray(scl, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    col0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    col1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    col2 = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    cols = np.stack([col0, col1, col2], axis=1)
    axis = np.argmin(s, axis=1)
    n = cols[np.arange(cols.shape[0]), axis, :]
    return (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


class RelightPanel(Panel):
    idname = "lfs.light_aware_relight"
    label = "Light-Aware GS"
    space = "SIDE_PANEL"
    order = 50

    def __init__(self):
        self.albedo = None; self.normals = None
        self.az, self.el, self.intensity, self.ambient = 30.0, 40.0, 1.2, 0.15
        self.color = (1.0, 1.0, 1.0); self.flip, self.live = False, True
        self._last = None; self._msg = ""
        self.proc = None; self.init_loaded = False; self._alb_mtime = 0.0
        self.prog = {}; self.iters = 300; self._session = False

    # ---- scene ----
    def _target(self):
        scene = lf.get_scene()
        if scene is None: return None
        found = None
        try:
            n = scene.get_node(NODE)
            if n is not None and getattr(n, "gaussian_count", 0) > 0:
                found = n.splat_data()
        except Exception: pass
        if found is None:
            try:
                for n in scene.get_nodes():                            # any node that carries gaussians
                    if getattr(n, "gaussian_count", 0) and hasattr(n, "splat_data") and n.splat_data() is not None:
                        found = n.splat_data(); break
            except Exception as e:
                _dbg(f"get_nodes threw: {e}")
        if found is None:
            try: found = scene.combined_model()                       # fallback: merged model
            except Exception: pass
        return found

    # ---- live training ----
    def _start(self):
        for f in (INIT_PLY, ALBEDO_NPY, PROGRESS):
            try: os.remove(f)
            except OSError: pass
        os.makedirs(LIVE, exist_ok=True)
        log = open(os.path.join(LIVE, "recover.log"), "w")
        self.proc = subprocess.Popen([FC_PY, RECOVER, "bearPNG", str(self.iters)], cwd=_REPO, stdout=log, stderr=subprocess.STDOUT)
        self.init_loaded = False; self._alb_mtime = 0.0
        self.prog = {"stage": "starting", "iter": 0, "total": self.iters, "loss": 0.0}
        self._msg = "training... grey bear will appear, then sharpen into albedo"
        try: open(DBG, "w").close()                                    # reset error log each run
        except Exception: pass
        self._session = True; self.init_loaded = False; self._alb_mtime = 0.0
        lf.on_frame(self._frame)                                       # backup driver (draw() is primary; LF clears this on any error)

    def _stop(self):
        if self.proc is not None:
            try: self.proc.terminate()
            except Exception: pass
            self.proc = None; self._msg = "training stopped"

    def _frame(self, dt=0.0):
        """Streaming poll; driven by draw() (reliable while panel open) + on_frame (backup). NEVER raise."""
        if not self._session: return
        try:
            if os.path.isfile(PROGRESS):
                try: self.prog = json.load(open(PROGRESS))
                except (ValueError, OSError): pass
            if not self.init_loaded and os.path.isfile(INIT_PLY):
                try: lf.clear_scene()                                  # own the scene: our bear must be the only splat, else it gets clobbered
                except Exception: pass
                lf.load_file(INIT_PLY); self.init_loaded = True
            if self.init_loaded and os.path.isfile(ALBEDO_NPY):
                m = os.path.getmtime(ALBEDO_NPY)
                if m != self._alb_mtime:
                    sd = self._target()
                    if sd is not None:
                        a = np.load(ALBEDO_NPY)
                        if a.shape[0] == sd.num_points:
                            _set_colors(sd, a)
                            self._alb_mtime = m                       # advance only on success -> retry until splat is ready
            if self.proc is not None and self.proc.poll() is not None:
                self.proc = None; self._msg = "recovery done - Capture albedo, then relight"
        except Exception as e:
            _dbg(f"_frame error: {e}\n{traceback.format_exc()}")

    # ---- relight ----
    def _capture(self, sd):
        self.albedo = np.asarray(sd.get_colors_rgb().cpu().numpy(), dtype=np.float32).copy()
        self.normals = _normals_from_gaussians(sd.get_rotation().cpu().numpy(), sd.get_scaling().cpu().numpy())
        self._last = None; self._msg = f"captured {self.albedo.shape[0]:,} Gaussians"

    def _relight(self, sd):
        el, az = math.radians(self.el), math.radians(self.az)
        l = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)], np.float32)
        n = -self.normals if self.flip else self.normals
        ndl = np.clip(n @ l, 0.0, None)[:, None]
        lc = np.array(self.color, np.float32)[None, :]
        col = np.clip(self.albedo * (self.ambient + self.intensity * ndl * lc), 0.0, 1.0)
        _set_colors(sd, col)

    # ---- UI (status + buttons only; streaming is in _frame) ----
    def draw(self, layout):
        try:
            self._draw(layout)
        except Exception as e:
            layout.text_colored(f"error: {e}", WARN)

    def _draw(self, layout):
        if self._session:
            self._frame()                                             # drive streaming from draw() -- fires reliably while panel is open
        layout.text_colored("Light-Aware GS", TITLE); layout.spacing()
        layout.text_colored("Pipeline", DIM)
        if layout.button("1. Load bear dataset", (-1, 0)):
            if os.path.isdir(DATASET):
                lf.load_file(DATASET, is_dataset=True); self._msg = "dataset loaded"
            else: self._msg = "missing data/lfs_bear"

        if self.proc is None:
            if not os.path.isfile(FC_PY):
                layout.text_colored("fullcircle python not found", WARN)
            elif layout.button("2. Recover albedo (LIVE train)", (-1, 0)):
                self._start()
        else:
            p = self.prog
            layout.text_colored(f"{p.get('stage','')} {p.get('iter',0)}/{p.get('total',0)}  L1 {p.get('loss',0):.4f}", WARN)
            if layout.button("Stop training", (-1, 0)): self._stop()

        if self._msg: layout.text_colored(self._msg, OK)
        layout.separator()

        layout.text_colored("Relight", DIM)
        sd = self._target()
        if sd is None:
            layout.text_colored("Recover or load a splat first.", DIM); return
        if layout.button("Capture albedo", (-1, 0)): self._capture(sd)
        if self.albedo is None:
            layout.text_colored("Snapshot current colors as material.", DIM); return
        layout.text_colored(f"material: {self.albedo.shape[0]:,} Gaussians", DIM)
        c1, self.az = layout.slider_float("Azimuth", self.az, -180.0, 180.0)
        c2, self.el = layout.slider_float("Elevation", self.el, -90.0, 90.0)
        c3, self.intensity = layout.slider_float("Intensity", self.intensity, 0.0, 4.0)
        c4, self.ambient = layout.slider_float("Ambient", self.ambient, 0.0, 1.0)
        c5, self.color = layout.color_edit3("Light color", self.color)
        c6, self.flip = layout.checkbox("Flip normals", self.flip)
        c7, self.live = layout.checkbox("Live relight", self.live)
        layout.spacing()
        params = (self.az, self.el, self.intensity, self.ambient, self.color, self.flip)
        changed = any([c1, c2, c3, c4, c5, c6]) or params != self._last
        if layout.button("Apply relight", (-1, 0)) or (self.live and changed):
            self._relight(sd); self._last = params
        if layout.button("Restore albedo", (-1, 0)):
            _set_colors(sd, self.albedo); self._last = None
