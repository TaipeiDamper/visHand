import random
import time

import cv2
import numpy as np


class FallingObject:
    SHAPES = ["circle", "rect", "triangle", "diamond", "pentagon", "hexagon"]
    SIDES_BY_SHAPE = {"triangle": 3, "diamond": 4, "pentagon": 5, "hexagon": 6}

    def __init__(self, w: int, h: int):
        self.size = random.randint(28, 62)
        self.x = random.randint(self.size, w - self.size)
        self.y = -self.size
        self.vx = random.uniform(-1.2, 1.2)
        self.vy = random.uniform(1.8, 4.8)
        self.angle = random.uniform(0.0, np.pi * 2.0)
        self.angular_v = random.uniform(-0.08, 0.08)
        self.spin_damping = random.uniform(0.98, 0.994)
        self.shape_type = random.choice(self.SHAPES)
        self.active = True
        self.color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
        self.hit = False
        self.hit_cooldown = 0
        self.restitution = random.uniform(0.68, 0.86)
        self.drag = random.uniform(0.984, 0.992)
        self.mass = max(1.0, (self.size / 24.0) ** 2)

    @property
    def radius(self) -> float:
        return self.size * 0.5

    def _polygon_points(self, sides: int, scale: float = 1.0):
        r = self.radius * scale
        pts = []
        for i in range(sides):
            a = self.angle + (2.0 * np.pi * i / sides)
            pts.append([int(self.x + np.cos(a) * r), int(self.y + np.sin(a) * r)])
        return np.array(pts, np.int32)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.angle += self.angular_v
        self.angular_v *= self.spin_damping
        self.vx *= self.drag
        self.vy *= self.drag
        if self.hit:
            self.vy += 0.23
        if self.hit_cooldown > 0:
            self.hit_cooldown -= 1

    def bounce_world_bounds(self, w: int, h: int):
        r = self.radius
        if self.x < r:
            self.x = r
            self.vx = abs(self.vx) * self.restitution
            self.angular_v += 0.02
        elif self.x > w - r:
            self.x = w - r
            self.vx = -abs(self.vx) * self.restitution
            self.angular_v -= 0.02

        if self.y < r:
            self.y = r
            self.vy = abs(self.vy) * self.restitution

        # Bottom edge keeps objects in play (more interaction time).
        if self.y > h - r:
            self.y = h - r
            self.vy = -abs(self.vy) * (self.restitution * 0.92)
            self.vx *= 0.97

    def draw(self, frame):
        if not self.active:
            return
        if self.shape_type == "circle":
            center = (int(self.x), int(self.y))
            r = int(self.radius)
            cv2.circle(frame, center, r, self.color, -1)
            cv2.circle(frame, center, r, (255, 255, 255), 2)
            mark = (int(self.x + np.cos(self.angle) * r * 0.8), int(self.y + np.sin(self.angle) * r * 0.8))
            cv2.line(frame, center, mark, (255, 255, 255), 2)
            return

        if self.shape_type == "rect":
            pts = self._polygon_points(4)
        else:
            pts = self._polygon_points(self.SIDES_BY_SHAPE[self.shape_type])
        cv2.fillPoly(frame, [pts], self.color)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)


class Minigame:
    def __init__(self):
        self.is_active = False
        self.start_hold_time = None
        self.exit_hold_time = None
        self.score = 0
        self.boxes: list[FallingObject] = []
        self.last_spawn = time.time()
        self.immunity_until_by_side = {}
        self.recent_fist_sides = {}
        self.curr_fist_sides = set()
        self.snap_cooldown_until = 0.0

    @staticmethod
    def _anchor_xy(payload: dict, w: int, h: int):
        anchor = payload["transform"]["anchor"]
        return anchor["x"] * w, anchor["y"] * h

    def _valid_indices(self, payloads: list[dict]):
        return [i for i, p in enumerate(payloads) if p and p["state"]["logic"] in ("ACTIVE", "HOVER")]

    def _handle_start_state(self, vis, payloads, valid_indices, timestamp, w, h):
        fists = sum(1 for p in payloads if p and p["state"]["intent"] == "CLOSED_FIST")
        if fists >= 2:
            if self.start_hold_time is None:
                self.start_hold_time = timestamp
            elif timestamp - self.start_hold_time > 0.8:
                self.is_active = True
                self.score = 0
                self.boxes = []
                self.start_hold_time = None
                self.exit_hold_time = None
                self.immunity_until_by_side.clear()
                self.recent_fist_sides.clear()
                cv2.putText(vis, "GAME START!", (w // 2 - 110, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        else:
            self.start_hold_time = None

    def _handle_exit_state(self, vis, valid_indices, raw_lms, timestamp, w, h):
        if len(valid_indices) != 2:
            self.exit_hold_time = None
            return False
        i1, i2 = valid_indices[0], valid_indices[1]
        lm1, lm2 = raw_lms[i1], raw_lms[i2]
        if not lm1 or not lm2:
            self.exit_hold_time = None
            return False

        t1, t2 = lm1.landmark[4], lm2.landmark[4]
        i1_lm, i2_lm = lm1.landmark[8], lm2.landmark[8]
        d_thumb = ((t1.x - t2.x) ** 2 + (t1.y - t2.y) ** 2) ** 0.5
        d_index = ((i1_lm.x - i2_lm.x) ** 2 + (i1_lm.y - i2_lm.y) ** 2) ** 0.5

        if d_thumb < 0.08 and d_index < 0.08:
            if self.exit_hold_time is None:
                self.exit_hold_time = timestamp
            elif timestamp - self.exit_hold_time > 1.0:
                self.is_active = False
                self.exit_hold_time = None
                cv2.putText(vis, "GAME END", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                return True
            else:
                progress = int((timestamp - self.exit_hold_time) / 1.0 * 100)
                cv2.putText(vis, f"Closing... {progress}%", (w // 2 - 100, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.exit_hold_time = None
        return False

    def _refresh_hand_states(self, payloads, valid_indices, timestamp):
        new_fist_sides = set()
        for idx in valid_indices:
            side = payloads[idx]["header"]["hand_side"]
            intent = payloads[idx]["state"]["intent"]
            if intent == "CLOSED_FIST":
                new_fist_sides.add(side)
                self.recent_fist_sides[side] = timestamp
        released_sides = self.curr_fist_sides - new_fist_sides
        for side in released_sides:
            self.immunity_until_by_side[side] = timestamp + 0.2
        self.curr_fist_sides = new_fist_sides

    def _spawn_objects(self, timestamp, w, h):
        interval = 1.25 if len(self.boxes) < 10 else 1.6
        if timestamp - self.last_spawn > interval:
            self.boxes.append(FallingObject(w, h))
            if random.random() < 0.18:
                self.boxes.append(FallingObject(w, h))
            self.last_spawn = timestamp

    def _apply_gesture_forces(self, payloads, valid_indices, timestamp, w, h):
        for idx in valid_indices:
            payload = payloads[idx]
            side = payload["header"]["hand_side"]
            intent = payload["state"]["intent"]
            event = payload["dynamics"]["event"]
            ax, ay = self._anchor_xy(payload, w, h)

            if event == "SNAP" and timestamp > self.snap_cooldown_until:
                self.snap_cooldown_until = timestamp + 0.1
                last_fist_time = self.recent_fist_sides.get(side, 0.0)
                if timestamp - last_fist_time < 0.8:
                    for box in self.boxes:
                        rx, ry = box.x - ax, box.y - ay
                        dist = max(1.0, (rx * rx + ry * ry) ** 0.5)
                        power = min(85.0, 3600.0 / dist)
                        box.vx += (rx / dist) * power / box.mass
                        box.vy += (ry / dist) * power / box.mass
                        box.angular_v += random.uniform(-0.2, 0.2)
                else:
                    self.boxes.clear()
                continue

            for box in self.boxes:
                if not box.active:
                    continue
                rx, ry = ax - box.x, ay - box.y
                dist = max(10.0, (rx * rx + ry * ry) ** 0.5)
                nx, ny = rx / dist, ry / dist
                tx, ty = -ny, nx

                if intent == "CLOSED_FIST":
                    pull = min(7.0, 320.0 / dist)
                    swirl = pull * 0.9
                    box.vx += (nx * pull + tx * swirl) / box.mass
                    box.vy += (ny * pull + ty * swirl) / box.mass
                    box.hit = True
                elif intent in ("PINCH_DRAG", "PINCH_HOLD"):
                    repel = min(6.5, 260.0 / dist)
                    box.vx -= (nx * repel) / box.mass
                    box.vy -= (ny * repel) / box.mass
                    box.angular_v += (tx * 0.03)
                elif intent == "OPEN_PALM":
                    if dist < 150.0:
                        lift = (150.0 - dist) / 150.0
                        box.vy -= 0.9 * lift
                        box.vx += tx * 0.1

    def _apply_hand_collision(self, box, payloads, raw_lms, valid_indices, timestamp, w, h):
        if box.hit_cooldown != 0:
            return
        for idx in valid_indices:
            side = payloads[idx]["header"]["hand_side"]
            if side in self.curr_fist_sides or timestamp < self.immunity_until_by_side.get(side, 0):
                continue

            lm = raw_lms[idx]
            if not lm:
                continue

            hand_dx = payloads[idx]["transform"]["delta"]["dx"] * w * 0.7
            hand_dy = payloads[idx]["transform"]["delta"]["dy"] * h * 0.7

            for point in lm.landmark:
                px, py = point.x * w, point.y * h
                dx, dy = box.x - px, box.y - py
                dist = max(0.1, (dx * dx + dy * dy) ** 0.5)
                if dist < (box.radius + 10):
                    nx, ny = dx / dist, dy / dist
                    impulse = 22.0
                    box.vx = nx * impulse + hand_dx
                    box.vy = ny * impulse - 9.0 + hand_dy
                    box.angular_v += random.uniform(-0.15, 0.15)
                    if not box.hit:
                        self.score += 1
                    box.hit = True
                    box.hit_cooldown = 4
                    return

    def _resolve_object_collisions(self):
        n = len(self.boxes)
        for i in range(n):
            a = self.boxes[i]
            if not a.active:
                continue
            for j in range(i + 1, n):
                b = self.boxes[j]
                if not b.active:
                    continue
                dx = b.x - a.x
                dy = b.y - a.y
                dist = (dx * dx + dy * dy) ** 0.5
                min_dist = a.radius + b.radius
                if dist <= 0.001 or dist >= min_dist:
                    continue

                nx, ny = dx / dist, dy / dist
                overlap = min_dist - dist
                a.x -= nx * (overlap * 0.5)
                a.y -= ny * (overlap * 0.5)
                b.x += nx * (overlap * 0.5)
                b.y += ny * (overlap * 0.5)

                rvx = b.vx - a.vx
                rvy = b.vy - a.vy
                vel_norm = rvx * nx + rvy * ny
                if vel_norm > 0:
                    continue

                e = min(a.restitution, b.restitution)
                j_impulse = -(1.0 + e) * vel_norm
                j_impulse /= (1.0 / a.mass) + (1.0 / b.mass)
                ix, iy = j_impulse * nx, j_impulse * ny

                a.vx -= ix / a.mass
                a.vy -= iy / a.mass
                b.vx += ix / b.mass
                b.vy += iy / b.mass

                spin = min(0.08, j_impulse * 0.003)
                a.angular_v -= spin
                b.angular_v += spin

    def _draw_hud(self, vis, payloads, valid_indices, w, h):
        cv2.putText(vis, f"Score: {self.score}", (w - 220, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(vis, "TRIANGLE to END", (w - 220, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, "FIST=pull  PINCH=repel  PALM=lift", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 240, 255), 2)

        for idx in valid_indices:
            side = payloads[idx]["header"]["hand_side"]
            intent = payloads[idx]["state"]["intent"]
            ax, ay = self._anchor_xy(payloads[idx], w, h)
            ax, ay = int(ax), int(ay)
            if intent == "CLOSED_FIST":
                cv2.circle(vis, (ax, ay), 58, (0, 0, 255), 2)
                cv2.putText(vis, "GRAVITY", (ax - 42, ay - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            elif intent in ("PINCH_DRAG", "PINCH_HOLD"):
                cv2.circle(vis, (ax, ay), 48, (255, 180, 0), 2)
                cv2.putText(vis, "REPEL", (ax - 30, ay - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
            elif intent == "OPEN_PALM":
                cv2.circle(vis, (ax, ay), 52, (80, 255, 180), 2)
                cv2.putText(vis, "LIFT", (ax - 22, ay - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 180), 2)

            if side in self.curr_fist_sides:
                cv2.circle(vis, (ax, ay), 72, (0, 0, 140), 1)

    def update(self, frame: np.ndarray, payloads: list[dict], raw_lms: list, timestamp: float) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = frame.copy()
        valid_indices = self._valid_indices(payloads)

        if not self.is_active:
            self._handle_start_state(vis, payloads, valid_indices, timestamp, w, h)
            msg = "Hold FIST with BOTH hands for 0.8s to TEST MODE"
            cv2.putText(vis, msg, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            if self.start_hold_time is not None:
                progress = int((timestamp - self.start_hold_time) / 0.8 * 100)
                cv2.putText(vis, f"Starting... {progress}%", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            return vis

        if self._handle_exit_state(vis, valid_indices, raw_lms, timestamp, w, h):
            return vis

        self._refresh_hand_states(payloads, valid_indices, timestamp)
        self._spawn_objects(timestamp, w, h)
        self._apply_gesture_forces(payloads, valid_indices, timestamp, w, h)

        for box in self.boxes:
            if not box.active:
                continue
            box.update()
            box.bounce_world_bounds(w, h)
            self._apply_hand_collision(box, payloads, raw_lms, valid_indices, timestamp, w, h)

        self._resolve_object_collisions()

        for box in self.boxes:
            box.draw(vis)
            if box.y > h + box.size * 2 or box.x < -box.size * 2 or box.x > w + box.size * 2:
                box.active = False
        self.boxes = [b for b in self.boxes if b.active]

        self._draw_hud(vis, payloads, valid_indices, w, h)
        return vis
