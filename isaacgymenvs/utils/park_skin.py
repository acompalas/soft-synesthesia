"""Park EGaIn pressure-skin model (Park et al. 2010 Eq. 1 = Park et al. 2012 Eq. 4) applied per cell.

For a channel of width w and height h under a uniform pressure p (position correction chi = 1):
    dR / R0 = 1 / (1 - p / p_star) - 1,        p_star = E h / (2 (1 - nu^2) w)
Each pad is split into cells (rigid bodies in the split robot). For cell k with compressive normal
force F_k and face area A_k:
    p_k = F_k / A_k,   x_k = min(p_k / (p_star * scale), alpha_cell),   g_k = 1 / (1 - x_k) - 1
The pad is ONE channel running through its cells, so the cells' resistances add in series: with
R0/n per cell, the pad's dR/R0 is the mean of the g_k. The readout saturates at g_max:
    reading = min(mean_k g_k, g_max) / g_max  in [0, 1],      g_max = 1 / (1 - alpha) - 1
(the environment clips every policy input to +-5, and binary touch is 0/1).
alpha_cell is set so that one cell alone reaching closure saturates the pad (g_k = n * g_max):
a channel closed anywhere reads as closed, whatever the loaded area. Under uniform pressure all
g_k are equal, so the pad follows Eq. 4 and saturates at x = alpha, as before.

Readout for the policy (log_eps): linear, reading = g_pad / g_max, or logarithmic,
    reading = log(1 + g_pad / eps) / log(1 + g_max / eps),
where eps is the smallest resolvable dR/R0. Both are monotonic in g_pad (same information, both in
[0, 1]); the log readout keeps typical touches visible next to saturated ones after the policy's
input normalisation. dR_from_reading() inverts either readout.

Defaults are Park et al. 2012: EcoFlex 00-30, E = 69 kPa; nu = 0.5 (assumed in the paper);
channels 200 um wide x 300 um high (all layers)  ->  p_star = 69 kPa.

This file does not import Isaac Gym, so the model can be tested on its own (test_park_skin.py).
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

import torch


def cell_areas_from_urdf(urdf_path, cell_names):
    """Face area (m^2) of each named cell: product of the two largest box dimensions
    (cells are thin along one axis)."""
    links = {l.get("name"): l for l in ET.parse(urdf_path).getroot().findall("link")}
    areas = []
    for name in cell_names:
        box = links[name].find("collision").find("geometry").find("box")
        dims = sorted(float(v) for v in box.get("size").split())
        areas.append(dims[1] * dims[2])
    return areas


class ParkSkin:
    def __init__(self, cell_areas, n_cells=4, E=69e3, nu=0.5, w=200e-6, h=300e-6, alpha=0.9,
                 log_eps=None, device="cpu"):
        self.n_cells = int(n_cells)
        self.area = torch.as_tensor(cell_areas, dtype=torch.float32, device=device)   # (16 * n_cells,)
        assert self.area.numel() % self.n_cells == 0
        self.n_pads = self.area.numel() // self.n_cells
        self.p_star = E * h / (2.0 * (1.0 - nu ** 2) * w)                              # Pa
        assert 0.0 < alpha < 1.0
        self.alpha = float(alpha)
        self.g_max = 1.0 / (1.0 - self.alpha) - 1.0                                   # pad readout cap
        self.g_cell_max = self.n_cells * self.g_max                                     # one closed cell saturates the pad
        self.alpha_cell = 1.0 - 1.0 / (self.g_cell_max + 1.0)
        self.log_eps = None if log_eps is None else float(log_eps)

    def cell_response(self, f_normal, p_star_scale=None):
        """f_normal: (N, 16 * n_cells) compressive normal force per cell, N.
        p_star_scale: optional (N, 16) per-pad multiplier on p_star (calibration variation).
        Returns raw dR/R0 per cell, (N, 16 * n_cells)."""
        p = torch.clamp(f_normal, min=0.0) / self.area
        p_star = torch.full_like(p, self.p_star)
        if p_star_scale is not None:
            p_star = p_star * p_star_scale.repeat_interleave(self.n_cells, dim=1)
        x = torch.clamp(p / p_star, 0.0, self.alpha_cell)
        return 1.0 / (1.0 - x) - 1.0

    def pad_readings(self, f_normal, p_star_scale=None):
        """(N, 16) pad readings in [0, 1]: cells in series (mean of their dR/R0), capped at g_max, then
        the readout (linear: / g_max; log: log(1 + g / eps) / log(1 + g_max / eps))."""
        g = self.cell_response(f_normal, p_star_scale)
        g_pad = torch.clamp(g.reshape(g.shape[0], self.n_pads, self.n_cells).mean(dim=-1), max=self.g_max)
        if self.log_eps is None:
            return g_pad / self.g_max
        return torch.log1p(g_pad / self.log_eps) / math.log1p(self.g_max / self.log_eps)

    def dR_from_reading(self, reading):
        """Pad dR/R0 from readings (inverse of the readout); works on numpy arrays or tensors."""
        if self.log_eps is None:
            return reading * self.g_max
        lib = torch if torch.is_tensor(reading) else np
        return self.log_eps * lib.expm1(reading * math.log1p(self.g_max / self.log_eps))
