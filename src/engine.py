"""物理引擎 - 带6个bug"""
from dataclasses import dataclass, field

@dataclass
class Body:
    pos: list
    vel: list
    size: list
    mass: float = 1.0
    sleeping: bool = False

class PhysicsEngine:
    def __init__(self):
        self.bodies: list[Body] = []
        self.gravity = [0, -9.8]
        self.contacts: list = []
        self.iterations = 1  # bug2: 只迭代一次，堆叠不稳定

    def add_body(self, body: Body):
        self.bodies.append(body)

    def check_continuous_collision(self, b1: Body, b2: Body, dt: float) -> bool:
        # bug1: 离散检测，高速穿墙
        return self._aabb_overlap(b1, b2)

    def _aabb_overlap(self, b1, b2):
        return (abs(b1.pos[0] - b2.pos[0]) < (b1.size[0] + b2.size[0]) / 2 and
                abs(b1.pos[1] - b2.pos[1]) < (b1.size[1] + b2.size[1]) / 2)

    def solve_contacts(self):
        # bug2: 单次迭代，不收敛
        for c in self.contacts:
            self._resolve_contact(c)

    def _resolve_contact(self, contact):
        b1, b2, normal, depth = contact
        # bug3: 摩擦用平均值，不是库仑模型
        friction = (b1.mass + b2.mass) / 2
        b1.vel[0] += normal[0] * depth * 0.5
        b2.vel[0] -= normal[0] * depth * 0.5
        b1.vel[1] += normal[1] * depth * 0.5
        b2.vel[1] -= normal[1] * depth * 0.5

    def update_joint(self, joint, angle):
        # bug4: 硬卡极限，无软约束
        if angle > joint["max"]:
            angle = joint["max"]
        elif angle < joint["min"]:
            angle = joint["min"]
        return angle

    def can_sleep(self, body: Body) -> bool:
        # bug5: 只看速度，不看加速度
        speed = (body.vel[0]**2 + body.vel[1]**2)**0.5
        return speed < 0.1

    def get_collision_force(self, contact) -> float:
        # bug6: 冲量计算前就返回0
        return 0.0

    def step(self, dt: float):
        for b in self.bodies:
            if self.can_sleep(b):
                b.sleeping = True
                continue
            b.vel[0] += self.gravity[0] * dt
            b.vel[1] += self.gravity[1] * dt
            b.pos[0] += b.vel[0] * dt
            b.pos[1] += b.vel[1] * dt
        self.contacts = []
        for i, b1 in enumerate(self.bodies):
            for b2 in self.bodies[i+1:]:
                if self.check_continuous_collision(b1, b2, dt):
                    self.contacts.append((b1, b2, [0, 1], 0.1))
        self.solve_contacts()
