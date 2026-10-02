"""物理引擎"""
import math
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
    restitution: float = 0.5
    sleeping: bool = False

class PhysicsEngine:
    MAX_RESTITUTION = 1.0   # 弹性系数上限，防止越弹越高
    SLEEP_THRESHOLD = 0.01  # 低于此速度才允许睡眠
    WAKE_THRESHOLD = 0.05   # 唤醒阈值调低，轻推即醒

    def __init__(self):
        self.bodies: dict[str, RigidBody] = {}

    @staticmethod
    def _corners(body: RigidBody) -> list[tuple[float, float]]:
        cx = body.x + body.width / 2
        cy = body.y + body.height / 2
        hw, hh = body.width / 2, body.height / 2
        rad = math.radians(body.rotation)
        c, s = math.cos(rad), math.sin(rad)
        return [
            (cx + dx * c - dy * s, cy + dx * s + dy * c)
            for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ]

    def check_collision(self, a: RigidBody, b: RigidBody) -> bool:
        # OBB碰撞检测：SAT分离轴定理，支持任意旋转
        corners_a = self._corners(a)
        corners_b = self._corners(b)
        for corners in (corners_a, corners_b):
            for i in range(len(corners)):
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % len(corners)]
                nx, ny = -(y2 - y1), (x2 - x1)
                norm = math.hypot(nx, ny)
                if norm == 0:
                    continue
                nx, ny = nx / norm, ny / norm
                proj_a = [px * nx + py * ny for px, py in corners_a]
                proj_b = [px * nx + py * ny for px, py in corners_b]
                if max(proj_a) < min(proj_b) or max(proj_b) < min(proj_a):
                    return False
        return True

    def sweep_collision(self, body: RigidBody, dx: float, dy: float) -> bool:
        # 连续碰撞检测：单帧位移超过自身尺寸即有穿透风险
        dist = math.hypot(dx, dy)
        min_dim = min(body.width, body.height)
        if dist > min_dim:
            return True
        # 沿扫掠路径分步采样，检测路径中间的碰撞
        steps = max(1, math.ceil(dist / (min_dim or 1.0)))
        for other in self.bodies.values():
            if other.bid == body.bid:
                continue
            for i in range(1, steps + 1):
                t = i / steps
                probe = RigidBody(body.bid, x=body.x + dx * t, y=body.y + dy * t,
                                  width=body.width, height=body.height,
                                  rotation=body.rotation)
                if self.check_collision(probe, other):
                    return True
        return False

    def apply_friction(self, body: RigidBody, friction: float, delta: float) -> None:
        # 摩擦力按时间持续衰减，而不是只应用一次
        factor = max(0.0, 1.0 - friction) ** delta
        body.vx *= factor
        body.vy *= factor

    def apply_physics_material(self, body: RigidBody, friction: float,
                               restitution: float, delta: float) -> None:
        self.apply_friction(body, friction, delta)
        # 弹性系数限制在[0, 1]，防止越弹越高
        body.restitution = min(max(restitution, 0.0), self.MAX_RESTITUTION)

    def sleep_body(self, body: RigidBody) -> None:
        # 睡眠中的物体被碰撞/交互时唤醒
        if body.sleeping:
            body.sleeping = False
            return
        speed = math.hypot(body.vx, body.vy)
        body.sleeping = speed < self.SLEEP_THRESHOLD

    def resolve_collision(self, a: RigidBody, b: RigidBody) -> None:
        # 碰撞会唤醒睡眠物体
        a.sleeping = False
        b.sleeping = False
        acx, acy = a.x + a.width / 2, a.y + a.height / 2
        bcx, bcy = b.x + b.width / 2, b.y + b.height / 2
        dx, dy = bcx - acx, bcy - acy
        overlap_x = (a.width + b.width) / 2 - abs(dx)
        overlap_y = (a.height + b.height) / 2 - abs(dy)
        if overlap_x <= 0 and overlap_y <= 0:
            return
        # 沿穿透最小的轴取碰撞法线
        if overlap_x < overlap_y:
            nx, ny = (1.0 if dx >= 0 else -1.0), 0.0
            penetration = overlap_x
        else:
            nx, ny = 0.0, (1.0 if dy >= 0 else -1.0)
            penetration = overlap_y
        # 位置修正：沿法线方向分离
        a.x -= nx * penetration / 2
        a.y -= ny * penetration / 2
        b.x += nx * penetration / 2
        b.y += ny * penetration / 2
        # 速度修正：沿法线施加冲量（等质量）
        rel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny
        if rel > 0:
            e = min(max(max(a.restitution, b.restitution), 0.0), self.MAX_RESTITUTION)
            j = (1 + e) * rel / 2
            a.vx -= j * nx
            a.vy -= j * ny
            b.vx += j * nx
            b.vy += j * ny
