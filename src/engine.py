"""物理引擎核心"""
import math
from dataclasses import dataclass, field


@dataclass
class Body:
    pos: list
    vel: list
    size: list
    mass: float = 1.0
    sleeping: bool = False
    friction_static: float = 0.6
    friction_kinetic: float = 0.4
    force: list = field(default_factory=lambda: [0.0, 0.0])


class PhysicsEngine:
    def __init__(self):
        self.bodies: list[Body] = []
        self.gravity = [0.0, -9.8]
        self.contacts: list = []
        self.iterations = 10            # fix2: 接触解算最大迭代数
        self.residual_threshold = 1e-4  # fix2: 残差低于该阈值即收敛
        self.contact_slop = 0.01        # 接触容差
        self.contact_stiffness = 10.0   # 碰撞力刚度
        self.ccd_speed_factor = 0.5     # fix1: 单步位移超过最小尺寸该比例才启用CCD
        self.sleep_speed_threshold = 0.1
        self.sleep_accel_threshold = 0.1
        self.collision_callbacks: list = []

    def add_body(self, body: Body):
        self.bodies.append(body)

    # ---- fix1: 扫掠体连续碰撞检测，仅对高速物体启用 ----
    def check_continuous_collision(self, b1: Body, b2: Body, dt: float) -> bool:
        if self._aabb_overlap(b1, b2):
            return True
        rvx = b1.vel[0] - b2.vel[0]
        rvy = b1.vel[1] - b2.vel[1]
        speed = math.hypot(rvx, rvy)
        min_size = min(b1.size[0], b1.size[1], b2.size[0], b2.size[1])
        if speed * dt < self.ccd_speed_factor * min_size:
            return False  # 低速物体用离散检测即可
        return self._swept_toi(b1, b2, dt) is not None

    def _swept_toi(self, b1: Body, b2: Body, dt: float):
        """扫掠AABB：相对运动线段与扩展盒(Minkowski和)求交，返回碰撞时刻t∈[0,1]或None"""
        rvx = b1.vel[0] - b2.vel[0]
        rvy = b1.vel[1] - b2.vel[1]
        # 相对运动起点（本步积分前的相对位置）
        sx = (b1.pos[0] - b2.pos[0]) - rvx * dt
        sy = (b1.pos[1] - b2.pos[1]) - rvy * dt
        hx = (b1.size[0] + b2.size[0]) / 2
        hy = (b1.size[1] + b2.size[1]) / 2
        t_enter, t_exit = 0.0, 1.0
        for s, d, h in ((sx, rvx * dt, hx), (sy, rvy * dt, hy)):
            if abs(d) < 1e-12:
                if abs(s) >= h:
                    return None
                continue
            t1 = (-h - s) / d
            t2 = (h - s) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
            if t_enter > t_exit:
                return None
        if t_exit < 0.0 or t_enter > 1.0:
            return None
        return max(t_enter, 0.0)

    def _aabb_overlap(self, b1, b2):
        return (abs(b1.pos[0] - b2.pos[0]) < (b1.size[0] + b2.size[0]) / 2 + self.contact_slop and
                abs(b1.pos[1] - b2.pos[1]) < (b1.size[1] + b2.size[1]) / 2 + self.contact_slop)

    def _compute_contact(self, b1, b2):
        dx = b1.pos[0] - b2.pos[0]
        dy = b1.pos[1] - b2.pos[1]
        ox = (b1.size[0] + b2.size[0]) / 2 + self.contact_slop - abs(dx)
        oy = (b1.size[1] + b2.size[1]) / 2 + self.contact_slop - abs(dy)
        if ox <= 0 or oy <= 0:
            return None
        if ox < oy:
            return (b1, b2, [1.0 if dx > 0 else -1.0, 0.0], ox)
        return (b1, b2, [0.0, 1.0 if dy > 0 else -1.0], oy)

    # ---- fix2: 多次迭代直到残差收敛或达到最大迭代数 ----
    def solve_contacts(self):
        for _ in range(max(1, self.iterations)):
            residual = 0.0
            for c in self.contacts:
                residual = max(residual, self._resolve_contact(c))
            if residual < self.residual_threshold:
                break
        # fix6: 冲量计算完成后才触发碰撞回调，并传入实际碰撞力
        for c in self.contacts:
            force = self.get_collision_force(c)
            for cb in self.collision_callbacks:
                cb(c, force)

    def _resolve_contact(self, contact):
        b1, b2, normal, depth = contact
        inv1 = 0.0 if b1.sleeping else 1.0 / b1.mass
        inv2 = 0.0 if b2.sleeping else 1.0 / b2.mass
        inv_sum = inv1 + inv2
        if inv_sum == 0.0:
            return 0.0
        # 法向冲量，按逆质量分配
        jn = depth * 0.5
        b1.vel[0] += normal[0] * jn * inv1 / inv_sum
        b1.vel[1] += normal[1] * jn * inv1 / inv_sum
        b2.vel[0] -= normal[0] * jn * inv2 / inv_sum
        b2.vel[1] -= normal[1] * jn * inv2 / inv_sum
        # fix3: 库仑摩擦模型，静摩擦 > 动摩擦
        tangent = [-normal[1], normal[0]]
        rv_t = ((b1.vel[0] - b2.vel[0]) * tangent[0] +
                (b1.vel[1] - b2.vel[1]) * tangent[1])
        mu_s = math.sqrt(b1.friction_static * b2.friction_static)
        mu_k = math.sqrt(b1.friction_kinetic * b2.friction_kinetic)
        jt = -rv_t  # 完全抵消切向相对速度所需的冲量
        if abs(jt) > mu_s * jn:
            # 静摩擦不足以保持相对静止，切换为动摩擦
            jt = math.copysign(mu_k * jn, jt)
        b1.vel[0] += tangent[0] * jt * inv1 / inv_sum
        b1.vel[1] += tangent[1] * jt * inv1 / inv_sum
        b2.vel[0] -= tangent[0] * jt * inv2 / inv_sum
        b2.vel[1] -= tangent[1] * jt * inv2 / inv_sum
        return abs(jn)

    # ---- fix4: 关节极限软约束，带弹性恢复 ----
    def update_joint(self, joint, angle):
        stiffness = joint.get("stiffness", 0.2)
        if angle > joint["max"]:
            return joint["max"] + (angle - joint["max"]) * stiffness
        if angle < joint["min"]:
            return joint["min"] + (angle - joint["min"]) * stiffness
        return angle

    # ---- fix5: 同时检查速度和加速度，持续受力（含重力）的物体不休眠 ----
    def can_sleep(self, body: Body) -> bool:
        speed = math.hypot(body.vel[0], body.vel[1])
        if speed >= self.sleep_speed_threshold:
            return False
        accel = math.hypot(
            self.gravity[0] + body.force[0] / body.mass,
            self.gravity[1] + body.force[1] / body.mass,
        )
        return accel < self.sleep_accel_threshold

    # ---- fix6: 根据接触计算实际碰撞冲量 ----
    def get_collision_force(self, contact) -> float:
        b1, b2, normal, depth = contact
        rvn = ((b1.vel[0] - b2.vel[0]) * normal[0] +
               (b1.vel[1] - b2.vel[1]) * normal[1])
        reduced_mass = (b1.mass * b2.mass) / (b1.mass + b2.mass)
        return reduced_mass * (abs(rvn) + depth * self.contact_stiffness)

    def step(self, dt: float):
        for b in self.bodies:
            if self.can_sleep(b):
                b.sleeping = True
                continue
            b.vel[0] += self.gravity[0] * dt
            b.vel[1] += self.gravity[1] * dt
            b.pos[0] += b.vel[0] * dt
            b.pos[1] += b.vel[1] * dt
        # fix1: 高速物体扫掠命中则回退到碰撞时刻，防止穿墙
        for i, b1 in enumerate(self.bodies):
            for b2 in self.bodies[i + 1:]:
                rvx = b1.vel[0] - b2.vel[0]
                rvy = b1.vel[1] - b2.vel[1]
                min_size = min(b1.size[0], b1.size[1], b2.size[0], b2.size[1])
                if math.hypot(rvx, rvy) * dt < self.ccd_speed_factor * min_size:
                    continue
                toi = self._swept_toi(b1, b2, dt)
                if toi is not None and 0.0 < toi < 1.0:
                    for b in (b1, b2):
                        b.pos[0] -= b.vel[0] * dt * (1.0 - toi)
                        b.pos[1] -= b.vel[1] * dt * (1.0 - toi)
        self.contacts = []
        for i, b1 in enumerate(self.bodies):
            for b2 in self.bodies[i + 1:]:
                if self.check_continuous_collision(b1, b2, dt):
                    contact = self._compute_contact(b1, b2)
                    if contact is not None:
                        self.contacts.append(contact)
        self.solve_contacts()
