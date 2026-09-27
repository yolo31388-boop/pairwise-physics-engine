"""物理引擎测试 - 8个失败"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import PhysicsEngine, Body

class TestCCD:
    def test_high_speed_no_tunneling(self):
        pe = PhysicsEngine()
        b1 = Body([0, 0], [100, 0], [1, 1])
        b2 = Body([50, 0], [0, 0], [1, 1])
        pe.add_body(b1)
        pe.add_body(b2)
        pe.step(0.016)
        assert not (b1.pos[0] > 51), f"高速穿墙: pos={b1.pos[0]}"

class TestStacking:
    def test_stack_stable_after_iterations(self):
        pe = PhysicsEngine()
        pe.iterations = 10
        b1 = Body([0, 0], [0, 0], [2, 2], mass=1)
        b2 = Body([0, 3], [0, 0], [2, 2], mass=1)
        pe.add_body(b1)
        pe.add_body(b2)
        for _ in range(5):
            pe.step(0.016)
        assert abs(b2.pos[1] - 3) < 0.5, f"堆叠不稳定: {b2.pos[1]}"

class TestFriction:
    def test_coulomb_friction(self):
        pe = PhysicsEngine()
        b1 = Body([0, 0], [10, 0], [1, 1], mass=1)
        b2 = Body([0, -1], [0, 0], [10, 1], mass=100)
        pe.add_body(b1)
        pe.add_body(b2)
        pe.step(0.1)
        assert b1.vel[0] < 10, "摩擦未生效"

class TestSleep:
    def test_sleep_checks_acceleration(self):
        pe = PhysicsEngine()
        b = Body([0, 0], [0, 0], [1, 1])
        b.vel = [0.05, 0.05]
        assert not pe.can_sleep(b), "持续受力物体不应休眠"

class TestCollisionForce:
    def test_collision_force_nonzero(self):
        pe = PhysicsEngine()
        contact = (Body([0,0],[0,0],[1,1]), Body([0,1],[0,0],[1,1]), [0,-1], 0.5)
        force = pe.get_collision_force(contact)
        assert force > 0, "碰撞力为0"
