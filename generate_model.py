"""Generate a physically symmetric 12-DOF radial tripod for MuJoCo.

Conventions: x points radially away from the body for each leg, z is up.
Each leg uses hip yaw (z), hip roll (x), hip pitch (y), knee pitch (y).
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tripod.xml"
LEG_NAMES = ("leg_a", "leg_b", "leg_c")
ANGLES_DEG = (0, 120, 240)
COLORS = ("0.22 0.67 0.94 1", "0.98 0.59 0.26 1", "0.48 0.82 0.45 1")


def child(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attrs.items()})


def generate_model(output: Path = MODEL_PATH) -> Path:
    root = ET.Element("mujoco", model="radial_tripod_12dof")
    child(root, "compiler", angle="degree", autolimits="true")
    child(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", solver="Newton", iterations="50")

    default = child(root, "default")
    child(default, "joint", type="hinge", damping="0.4", armature="0.003", limited="true")
    child(default, "geom", contype="1", conaffinity="1", friction="1.25 0.005 0.0001", condim="3")

    visual = child(root, "visual")
    # Offscreen framebuffer must be at least as large as video frames.
    # Insert within the existing <visual> tag, rather than adding a second tag.
    child(visual, "global", offwidth="1280", offheight="720")
    child(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.65 0.65 0.65", specular="0.08 0.08 0.08")
    child(visual, "map", znear="0.01")
    child(visual, "quality", shadowsize="2048")

    world = child(root, "worldbody")
    child(world, "light", pos="0 -1.5 2.8", dir="0 0 -1", directional="true")
    child(world, "geom", name="ground", type="plane", size="10 10 0.1", rgba="0.82 0.85 0.87 1", friction="1.3 0.005 0.0001")

    torso = child(world, "body", name="torso", pos="0 0 0.46")
    child(torso, "freejoint", name="root")
    child(torso, "geom", name="chassis", type="cylinder", pos="0 0 0", size="0.15 0.042", mass="1.20", rgba="0.18 0.23 0.34 1")
    child(torso, "geom", name="chassis_top", type="cylinder", pos="0 0 0.045", size="0.11 0.004", mass="0.04", rgba="0.32 0.39 0.51 1", contype="0", conaffinity="0")
    child(torso, "geom", name="forward_marker", type="box", pos="0.105 0 0.052", size="0.025 0.012 0.006", mass="0.004", rgba="0.96 0.96 0.96 1", contype="0", conaffinity="0")
    child(torso, "site", name="torso_center", size="0.01", rgba="1 1 1 1")

    actuator = child(root, "actuator")
    for name, angle, rgba in zip(LEG_NAMES, ANGLES_DEG, COLORS):
        radians = math.radians(angle)
        hip_pos = f"{0.159 * math.cos(radians):.7f} {0.159 * math.sin(radians):.7f} -0.015"
        yaw = child(torso, "body", name=f"{name}_hip_yaw", pos=hip_pos, euler=f"0 0 {angle}")
        child(yaw, "joint", name=f"{name}_yaw", axis="0 0 1", range="-55 55")
        child(yaw, "geom", type="sphere", size="0.026", mass="0.075", rgba=rgba, contype="0", conaffinity="0")

        roll = child(yaw, "body", name=f"{name}_hip_roll")
        child(roll, "joint", name=f"{name}_roll", axis="1 0 0", range="-45 45")
        child(roll, "geom", type="sphere", size="0.022", mass="0.055", rgba=rgba, contype="0", conaffinity="0")

        pitch = child(roll, "body", name=f"{name}_hip_pitch")
        child(pitch, "joint", name=f"{name}_pitch", axis="0 1 0", range="-75 75")
        child(pitch, "geom", name=f"{name}_thigh", type="capsule", fromto="0 0 0 0.06 0 -0.20", size="0.020", mass="0.21", rgba=rgba)

        knee = child(pitch, "body", name=f"{name}_knee_body", pos="0.06 0 -0.20")
        child(knee, "joint", name=f"{name}_knee", axis="0 1 0", range="0 115")
        child(knee, "geom", name=f"{name}_shin", type="capsule", fromto="0 0 0 0.03 0 -0.20", size="0.016", mass="0.16", rgba=rgba)
        child(knee, "geom", name=f"{name}_foot", type="sphere", pos="0.03 0 -0.20", size="0.040", mass="0.07", rgba="0.13 0.14 0.15 1", friction="1.55 0.006 0.0001")
        child(knee, "site", name=f"{name}_foot_tip", pos="0.03 0 -0.20", size="0.013", rgba=rgba)

        # Control range and servo target are both in radians, not degrees.
        for joint_suffix, kp, kv, force_limit, lo, hi in (
            ("yaw", 75, 5.5, 24, -0.9599, 0.9599),
            ("roll", 75, 5.5, 24, -0.7853, 0.7853),
            ("pitch", 85, 6.5, 30, -1.3089, 1.3089),
            ("knee", 85, 5.0, 30, 0.0, 2.0070),
        ):
            child(actuator, "position", name=f"{name}_{joint_suffix}_servo", joint=f"{name}_{joint_suffix}",
                  kp=str(kp), kv=str(kv), ctrllimited="true", ctrlrange=f"{lo} {hi}",
                  forcelimited="true", forcerange=f"{-force_limit} {force_limit}")

    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=True)
    return output


if __name__ == "__main__":
    print(f"Wrote {generate_model()}")
