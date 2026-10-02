"""物理引擎 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.engine import PhysicsEngine, RigidBody

class TestOBBCollision:
    def test_rotated_body_collision_detected(self):
        pe = PhysicsEngine()
        a = RigidBody("a", x=0, y=0, width=2, height=1, rotation=45)
        b = RigidBody("b", x=1, y=0, width=1, height=1)
        assert pe.check_collision(a, b) == True

class TestSweepCollision:
    def test_high_speed_body_does_not_tunnel(self):
        pe = PhysicsEngine()
        body = RigidBody("b", x=0, y=0, width=0.1, height=0.1)
        # 高速移动穿过障碍物
        result = pe.sweep_collision(body, 10, 0)
        assert result == True

class TestContinuousFriction:
    def test_friction_applied_continuously(self):
        pe = PhysicsEngine()
        body = RigidBody("b", vx=10)
        pe.apply_friction(body, 0.1, 0.5)
        assert body.vx < 10

class TestSleepingCollision:
    def test_sleeping_body_wakes_on_collision(self):
        pe = PhysicsEngine()
        body = RigidBody("b", sleeping=True)
        pe.sleep_body(body)
        # 被碰撞时应该唤醒
        assert body.sleeping == False

class TestVelocityResolution:
    def test_collision_resolves_velocity(self):
        pe = PhysicsEngine()
        a = RigidBody("a", x=0, vx=5)
        b = RigidBody("b", x=0.5, vx=0)
        pe.resolve_collision(a, b)
        assert a.vx < 5
