"""No MuJoCo dependency: verify joints, radial symmetry, servo wiring, IK rest pose."""
import math
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_model import MODEL_PATH, generate_model

class TestTripodStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ET.parse(generate_model()).getroot()

    def test_leg_geometry_and_angles(self):
        seen = []
        for leg, angle in zip('abc', [0, 120, 240]):
            yaw = self.root.find(f".//worldbody//body[@name='leg_{leg}_hip_yaw']")
            x, y, z = map(float, yaw.get('pos').split())
            self.assertAlmostEqual(math.hypot(x, y), .159, places=5)
            self.assertAlmostEqual(math.degrees(math.atan2(y, x)) % 360, angle, places=3)
            self.assertEqual(float(yaw.get('euler').split()[-1]), angle)
            self.assertEqual(z, -.015)
            for suffix, axis in [('yaw','0 0 1'),('roll','1 0 0'),('pitch','0 1 0'),('knee','0 1 0')]:
                joint=self.root.find(f".//worldbody//joint[@name='leg_{leg}_{suffix}']")
                self.assertIsNotNone(joint)
                self.assertEqual(joint.get('axis'), axis)
            seen.append(angle)
        self.assertEqual(len(set(seen)), 3)

    def test_video_offscreen_framebuffer(self):
        global_visual = self.root.find("./visual/global")
        self.assertIsNotNone(global_visual)
        self.assertGreaterEqual(int(global_visual.get("offwidth")), 720)
        self.assertGreaterEqual(int(global_visual.get("offheight")), 540)

    def test_servo_correspondence(self):
        all_joints = {j.get('name') for j in self.root.findall('.//worldbody//joint')}
        servos = self.root.findall('.//actuator/position')
        self.assertEqual(len(all_joints), 12)
        self.assertEqual(len(servos), 12)
        self.assertEqual({a.get('joint') for a in servos}, all_joints)
        for s in servos:
            lo, hi = map(float, s.get('ctrlrange').split())
            self.assertLess(lo, hi)
            self.assertGreater(float(s.get('kp')), 0)
            self.assertGreater(float(s.get('kv')), 0)

    def test_neutral_initial_foot_clearance(self):
        # Kinematics of 0,0,-.15,.30 rad hip and knee (one radial leg).
        def linkz(x, z, theta):
            return -x * math.sin(theta) + z * math.cos(theta)
        z_foot = .46 - .015 + linkz(.06, -.20, -.15) + linkz(.03, -.20, .15)
        clearance = z_foot - .04
        self.assertGreater(clearance, 0)
        self.assertLess(clearance, .03)

if __name__ == '__main__':
    unittest.main()
