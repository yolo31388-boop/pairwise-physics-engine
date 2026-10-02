"""物理引擎 - 修复版"""
import math
from dataclasses import dataclass

# 弹性系数上限，防止越弹越高
MAX_RESTITUTION = 1.0
# 睡眠/唤醒阈值（唤醒阈值调低，轻推即醒）
SLEEP_THRESHOLD = 0.05
WAKE_THRESHOLD = 0.1


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


def _corners(body: RigidBody) -> list[tuple[float, float]]:
    """返回旋转后包围盒(OBB)的四个顶点，绕中心旋转。"""
    cx = body.x + body.width / 2
    cy = body.y + body.height / 2
    hw = body.width / 2
    hh = body.height / 2
    rad = math.radians(body.rotation)
    c = math.cos(rad)
    s = math.sin(rad)
    return [
        (cx + dx * c - dy * s, cy + dx * s + dy * c)
        for dx, dy in ((hw, hh), (hw, -hh), (-hw, hh), (-hw, -hh))
    ]


class PhysicsEngine:
    def __init__(self):
        self.bodies: dict[str, RigidBody] = {}

    def check_collision(self, a: RigidBody, b: RigidBody) -> bool:
        # fix1: OBB旋转包围盒检测（SAT分离轴定理），旋转45度的箱子也能正确检测
        ca = _corners(a)
        cb = _corners(b)
        for corners in (ca, cb):
            for i in range(len(corners)):
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % len(corners)]
                axis_x = -(y2 - y1)
                axis_y = x2 - x1
                a_proj = [axis_x * px + axis_y * py for px, py in ca]
                b_proj = [axis_x * px + axis_y * py for px, py in cb]
                if max(a_proj) < min(b_proj) or max(b_proj) < min(a_proj):
                    return False
        return True

    def sweep_collision(self, body: RigidBody, dx: float, dy: float) -> bool:
        # fix2: 连续碰撞检测，沿扫掠路径分子步检测，不只看起点终点。
        # 单步位移超过自身尺寸时有穿透风险，返回True提示必须走CCD。
        distance = math.hypot(dx, dy)
        min_size = min(body.width, body.height)
        tunneling_risk = distance > min_size
        step_len = max(min_size / 2, 1e-6)
        steps = max(1, math.ceil(distance / step_len))
        for i in range(1, steps + 1):
            t = i / steps
            moved = RigidBody(
                body.bid,
                x=body.x + dx * t,
                y=body.y + dy * t,
                width=body.width,
                height=body.height,
                rotation=body.rotation,
            )
            for other in self.bodies.values():
                if other is not body and self.check_collision(moved, other):
                    return True
        return tunneling_risk

    def apply_friction(self, body: RigidBody, friction: float, delta: float) -> None:
        # fix3: 摩擦按时间步持续计算，持续接触时每一帧都减速
        decay = max(0.0, 1.0 - friction * delta)
        body.vx *= decay
        body.vy *= decay

    def sleep_body(self, body: RigidBody) -> None:
        # fix4: 睡眠物体仍参与碰撞检测，被碰撞或速度超过(调低后的)唤醒阈值时唤醒；
        # 只有引擎管理且静止无碰撞的物体才允许进入睡眠
        speed = math.hypot(body.vx, body.vy)
        colliding = any(
            self.check_collision(body, other)
            for other in self.bodies.values()
            if other is not body
        )
        if colliding or speed > WAKE_THRESHOLD:
            body.sleeping = False
        elif speed < SLEEP_THRESHOLD and body.bid in self.bodies:
            body.sleeping = True
        else:
            body.sleeping = False

    def resolve_collision(self, a: RigidBody, b: RigidBody, restitution: float = 0.5) -> None:
        # fix5: 同时修正位置和速度，沿碰撞法线分离，弹性系数限制在[0,1]
        a.sleeping = False
        b.sleeping = False
        e = max(0.0, min(MAX_RESTITUTION, restitution))

        acx = a.x + a.width / 2
        acy = a.y + a.height / 2
        bcx = b.x + b.width / 2
        bcy = b.y + b.height / 2
        overlap_x = (a.width + b.width) / 2 - abs(acx - bcx)
        overlap_y = (a.height + b.height) / 2 - abs(acy - bcy)
        if overlap_x <= 0 or overlap_y <= 0:
            return

        # 沿穿透最浅的轴分离，法线方向由b指向a
        if overlap_x < overlap_y:
            nx, ny = (1.0, 0.0) if acx >= bcx else (-1.0, 0.0)
            depth = overlap_x
        else:
            nx, ny = (0.0, 1.0) if acy >= bcy else (0.0, -1.0)
            depth = overlap_y

        # 位置修正：两物体各退一半，避免卡在碰撞体里抖动
        a.x += nx * depth / 2
        a.y += ny * depth / 2
        b.x -= nx * depth / 2
        b.y -= ny * depth / 2

        # 速度修正：沿法线方向施加冲量（等质量），只在相互接近时应用
        rel_vx = a.vx - b.vx
        rel_vy = a.vy - b.vy
        rel_normal = rel_vx * nx + rel_vy * ny
        if rel_normal < 0:
            impulse = -(1 + e) * rel_normal / 2
            a.vx += impulse * nx
            a.vy += impulse * ny
            b.vx -= impulse * nx
            b.vy -= impulse * ny
