"""物理引擎 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class RigidBody:
    bid: str
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    rotation: float = 0.0
    sleeping: bool = False

class PhysicsEngine:
    def __init__(self):
        self.bodies: dict[str, RigidBody] = {}

    def check_collision(self, a: RigidBody, b: RigidBody) -> bool:
        # bug1: 只检测AABB，不检测旋转
        return (a.x < b.x + b.width and a.x + a.width > b.x and
                a.y < b.y + b.height and a.y + a.height > b.y)

    def sweep_collision(self, body: RigidBody, dx: float, dy: float) -> bool:
        # bug2: 只看起点终点，不看路径
        new_x = body.x + dx
        new_y = body.y + dy
        return False

    def apply_friction(self, body: RigidBody, friction: float, delta: float) -> None:
        # bug3: 摩擦只应用一次
        body.vx *= (1 - friction)

    def sleep_body(self, body: RigidBody) -> None:
        # bug4: 睡眠物体不检测碰撞
        body.sleeping = True

    def resolve_collision(self, a: RigidBody, b: RigidBody) -> None:
        # bug5: 只修正位置不修正速度
        overlap_x = (a.width + b.width) / 2 - abs(a.x - b.x)
        if overlap_x > 0:
            a.x += overlap_x / 2
