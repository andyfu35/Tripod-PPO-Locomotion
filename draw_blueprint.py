"""Draw schematic design PNG/SVG; not a trained gait visualization."""
from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent

def draw():
    fig = plt.figure(figsize=(12, 5.5), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.1])
    a0, a1 = fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1])
    a0.add_patch(Circle((0,0), 0.15, facecolor="#384659", edgecolor="#1f2a39", linewidth=2, zorder=4))
    a0.add_patch(Circle((0,0), 0.11, facecolor="#5b6d81", edgecolor="none", zorder=5))
    palette = ("#2d93c7", "#ed9b46", "#64b278")
    for i,deg in enumerate((0,120,240)):
        th = math.radians(deg)
        u = np.array([math.cos(th), math.sin(th)])
        hip = 0.159*u
        knee = 0.248*u
        foot = 0.280*u
        a0.plot([hip[0],knee[0],foot[0]], [hip[1],knee[1],foot[1]], lw=9, color=palette[i], solid_capstyle="round", zorder=3)
        a0.plot(*hip, "o", markersize=9, color="white", markeredgecolor="#222", zorder=6)
        a0.add_patch(Circle(foot, 0.040, facecolor="#2c3036", edgecolor="white", zorder=5))
        lab = 0.39*u
        a0.text(*lab, f"Leg {'ABC'[i]}\n{deg} deg", ha="center", va="center", fontsize=11, fontweight="bold")
    a0.plot([0,0.11],[0,0],color="white",lw=3,zorder=7)
    a0.text(0, -0.06, "Circular torso", color="white", ha="center", zorder=8)
    a0.set_xlim(-0.48,0.48); a0.set_ylim(-0.45,0.45)
    a0.set_aspect("equal"); a0.axis("off"); a0.set_title("Top view - 120 deg radial symmetry", fontsize=14, pad=16)

    # Side-profile link construction for one leg, with neutral joint angles.
    hx, hz = 0.159, 0.46-0.015
    thigh_end = np.array([0.06, -0.20]); shin_end = np.array([0.03, -0.20])
    def ry(xz,th):
        return np.array([[math.cos(th),math.sin(th)],[-math.sin(th),math.cos(th)]]) @ xz
    knee = np.array([hx,hz]) + ry(thigh_end, -0.15)
    toe = knee + ry(shin_end, 0.15)
    a1.axhline(0,color="#888",lw=2)
    a1.add_patch(plt.Rectangle((-0.15,0.46-0.042), 0.30, 0.084, facecolor="#384659",edgecolor="white",lw=1))
    a1.plot([hx,knee[0]],[hz,knee[1]],color=palette[0],lw=13,solid_capstyle="round")
    a1.plot([knee[0],toe[0]],[knee[1],toe[1]],color="#5fb0e4",lw=10,solid_capstyle="round")
    for pos,lab,off in [(np.array([hx,hz]),"3-axis hip",(0.06,0.07)),(knee,"1-axis knee",(0.055,0.055))]:
        a1.plot(*pos,"o",color="#ffc857",markersize=11,zorder=5)
        a1.annotate(lab, xy=pos, xytext=pos+off, fontsize=11,arrowprops={"arrowstyle":"-", "lw":1})
    a1.add_patch(Circle(toe, 0.04,facecolor="#2c3036",edgecolor="black"))
    a1.annotate("Foot contact",xy=toe,xytext=(toe[0]+.07,toe[1]-.04),fontsize=11,arrowprops={"arrowstyle":"-", "lw":1})
    a1.annotate("0.46 m initial base height", xy=(-0.20, 0),xytext=(-0.20,.46),va="center",ha="right",fontsize=10,
                arrowprops={"arrowstyle":"<->", "lw":1.5})
    a1.set_xlim(-.40,.60);a1.set_ylim(-.06,.67);a1.set_aspect("equal");a1.axis("off")
    a1.set_title("Side view - 2 links, 4 DOF per leg",fontsize=14,pad=16)
    for ext in ("png","svg"):
        out=OUT / f"tripod_blueprint.{ext}"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        print(out)
    plt.close(fig)

if __name__ == "__main__":
    draw()
