"""Static XML validation plus optional MuJoCo/Gymnasium runtime checks."""
import ast
import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

from generate_model import MODEL_PATH, generate_model


def main():
    path = generate_model()
    root = ET.parse(path).getroot()
    assert root.attrib["model"] == "radial_tripod_12dof"
    assert len(root.findall(".//worldbody//joint")) == 12
    assert len(root.findall(".//freejoint")) == 1
    assert len(root.findall(".//actuator/position")) == 12
    assert len(root.findall(".//site[@name]")) >= 4
    for letter in "abc":
        assert root.find(f".//geom[@name='leg_{letter}_foot']") is not None
        assert root.find(f".//joint[@name='leg_{letter}_knee']") is not None
    for pyfile in Path(__file__).parent.glob("*.py"):
        ast.parse(pyfile.read_text(encoding="utf-8"), filename=str(pyfile))
    print("PASS: XML well-formed; 3 legs x 4 hinges; 12 servos; Python AST parses.")

    if not all(importlib.util.find_spec(m) for m in ("mujoco", "gymnasium")):
        print("SKIP: Cannot run physics here (mujoco/gymnasium not installed).")
        return
    import numpy as np
    import mujoco
    from tripod_env import TripodEnv
    model = mujoco.MjModel.from_xml_path(str(path))
    assert (model.nq, model.nv, model.nu) == (19, 18, 12)
    env = TripodEnv(max_episode_steps=100)
    obs, info = env.reset(seed=123)
    assert obs.shape == (51,) and np.isfinite(obs).all()
    for i in range(30):
        obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        assert obs.shape == (51,) and np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            break
    print(f"PASS: MuJoCo physics, action/observation, {i+1} simulation steps. Alive={not terminated}")
    if importlib.util.find_spec("stable_baselines3"):
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("PASS: Stable-Baselines3 Gymnasium interface check.")
    env.close()


if __name__ == "__main__":
    main()
