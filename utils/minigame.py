import time
import random
import cv2
import numpy as np

class FallingObject:
    def __init__(self, w, h):
        self.size = random.randint(30, 60)
        self.x = random.randint(self.size, w - self.size)
        self.y = -self.size
        # Fall downwards
        self.vy = random.uniform(2.0, 5.0)
        self.vx = random.uniform(-1.0, 1.0)
        self.active = True
        self.color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        self.hit = False
        self.hit_cooldown = 0
        self.shape_type = random.choice(["rect", "circle"])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Free fall gravity ONLY if it wasn't hit and isn't caught
        if self.hit:
            self.vy += 0.5  # Standard gravity after hit
            
        # Friction/Air resistance for smooth orbital movement
        self.vx *= 0.98
        self.vy *= 0.98
            
        if self.hit_cooldown > 0:
            self.hit_cooldown -= 1
        
    def draw(self, frame):
        if not self.active: return
        
        if self.shape_type == "rect":
            pts = np.array([
                [int(self.x - self.size/2), int(self.y - self.size/2)],
                [int(self.x + self.size/2), int(self.y - self.size/2)],
                [int(self.x + self.size/2), int(self.y + self.size/2)],
                [int(self.x - self.size/2), int(self.y + self.size/2)]
            ], np.int32)
            cv2.fillPoly(frame, [pts], self.color)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
        else:
            center = (int(self.x), int(self.y))
            r = int(self.size / 2)
            cv2.circle(frame, center, r, self.color, -1)
            cv2.circle(frame, center, r, (255, 255, 255), 2)


class Minigame:
    def __init__(self):
        self.is_active = False
        self.start_hold_time = None
        self.exit_hold_time = None
        self.score = 0
        self.boxes = []
        self.last_spawn = time.time()
        
        # Physics memory
        self.immunity_until_by_side = {}     # Mapping hand_side -> timestamp
        self.recent_fist_sides = {}          # Mapping hand_side -> timestamp when it was last a FIST
        self.curr_fist_sides = set()         # Set of strings "LEFT" / "RIGHT"

    def update(self, frame: np.ndarray, payloads: list[dict], raw_lms: list, timestamp: float) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = frame.copy()
        
        # Valid tracking logic hands
        valid_indices = [i for i, p in enumerate(payloads) if p and p["state"]["logic"] in ("ACTIVE", "HOVER")]
        
        # Check start condition: both FIST for 1.5s
        if not self.is_active and len(valid_indices) == 2:
            i1, i2 = valid_indices[0], valid_indices[1]
            if payloads[i1]["state"]["intent"] == "FIST" and payloads[i2]["state"]["intent"] == "FIST":
                if self.start_hold_time is None:
                    self.start_hold_time = timestamp
                elif timestamp - self.start_hold_time > 1.5:
                    self.is_active = True
                    self.score = 0
                    self.boxes = []
                    self.start_hold_time = None
                    self.exit_hold_time = None
                    self.immunity_until_by_side.clear()
                    self.recent_fist_sides.clear()
                    cv2.putText(vis, "GAME START!", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            else:
                self.start_hold_time = None
        else:
            self.start_hold_time = None

        if self.is_active:
            # Check stop condition: Triangle gesture (Thumbs and Indexes touching)
            if len(valid_indices) == 2:
                i1, i2 = valid_indices[0], valid_indices[1]
                lm1, lm2 = raw_lms[i1], raw_lms[i2]
                if lm1 and lm2:
                    t1, t2 = lm1.landmark[4], lm2.landmark[4]  # THUMB_TIP
                    i1_lm, i2_lm = lm1.landmark[8], lm2.landmark[8]  # INDEX_TIP
                    
                    d_thumb = ((t1.x - t2.x)**2 + (t1.y - t2.y)**2)**0.5
                    d_index = ((i1_lm.x - i2_lm.x)**2 + (i1_lm.y - i2_lm.y)**2)**0.5
                    
                    if d_thumb < 0.08 and d_index < 0.08:
                        if self.exit_hold_time is None:
                            self.exit_hold_time = timestamp
                        elif timestamp - self.exit_hold_time > 1.0:
                            self.is_active = False
                            self.exit_hold_time = None
                            cv2.putText(vis, "GAME END", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            return vis
                        else:
                            # Show triangle timer
                            progress = int((timestamp - self.exit_hold_time) / 1.0 * 100)
                            cv2.putText(vis, f"Closing... {progress}%", (w//2 - 100, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        self.exit_hold_time = None
                else:
                    self.exit_hold_time = None
            else:
                self.exit_hold_time = None

            # Track which sides are currently FISTs
            new_fist_sides = set()
            for idx in valid_indices:
                side = payloads[idx]["header"]["hand_side"]
                intent = payloads[idx]["state"]["intent"]
                if intent == "FIST":
                    new_fist_sides.add(side)
                    self.recent_fist_sides[side] = timestamp

            # Detect FIST releases -> Add Immunity
            released_sides = self.curr_fist_sides - new_fist_sides
            for side in released_sides:
                self.immunity_until_by_side[side] = timestamp + 0.2  # 0.2s collision immunity
            
            self.curr_fist_sides = new_fist_sides

            # Check Snap triggers from any hand
            for idx in valid_indices:
                event = payloads[idx]["dynamics"]["event"]
                side = payloads[idx]["header"]["hand_side"]
                if event == "SNAP":
                    # Was this hand a recent gravitational FIST?
                    last_fist_time = self.recent_fist_sides.get(side, 0)
                    if timestamp - last_fist_time < 0.8:
                        # Repel explosion (from this hand's anchor to all boxes)
                        anchor = payloads[idx]["transform"]["anchor"]
                        ax, ay = anchor["x"] * w, anchor["y"] * h
                        for box in self.boxes:
                            rx, ry = box.x - ax, box.y - ay
                            r_len = max(1.0, (rx**2 + ry**2)**0.5)
                            box.vx += (rx / r_len) * 60.0
                            box.vy += (ry / r_len) * 60.0
                    else:
                        # Clear screen
                        self.boxes.clear()

            # Spawn boxes
            if timestamp - self.last_spawn > 1.5:
                self.boxes.append(FallingObject(w, h))
                self.last_spawn = timestamp

            # Update and draw objects
            # Apply Gravity logic if FIST exists
            for box in self.boxes:
                if not box.active: continue

                # Gravity
                for idx in valid_indices:
                    side = payloads[idx]["header"]["hand_side"]
                    if side in self.curr_fist_sides:
                        # Draw towards anchor
                        anchor = payloads[idx]["transform"]["anchor"]
                        ax, ay = anchor["x"] * w, anchor["y"] * h
                        rx, ry = ax - box.x, ay - box.y
                        
                        dist = max(10.0, (rx**2 + ry**2)**0.5)
                        
                        # Scale force: gentle pull
                        pull_force = 300.0 / dist
                        pull_force = min(pull_force, 6.0)

                        # Normalized vector towards hand
                        nx, ny = rx / dist, ry / dist

                        # Tangential orbital push (perpendicular)
                        ox, oy = -ny, nx

                        # Apply forces
                        box.vx += nx * pull_force + ox * (pull_force * 1.2)
                        box.vy += ny * pull_force + oy * (pull_force * 1.2)
                        
                        # Stabilize orbit by capping speed and adding friction
                        speed = (box.vx**2 + box.vy**2)**0.5
                        max_speed = 20.0
                        if speed > max_speed:
                            box.vx = (box.vx / speed) * max_speed
                            box.vy = (box.vy / speed) * max_speed
                            
                        # Extra friction inside gravity well
                        box.vx *= 0.94
                        box.vy *= 0.94
                        
                        box.hit = True  # Ensures it's unbound from free-fall

                box.update()
                
                # Check collision with ANY node of ANY hand
                if box.hit_cooldown == 0:
                    collision_found = False
                    for idx in valid_indices:
                        if collision_found: break
                        
                        side = payloads[idx]["header"]["hand_side"]
                        # Check immunity and FIST status (bypass collision if it's currently attracting objects)
                        if side in self.curr_fist_sides or timestamp < self.immunity_until_by_side.get(side, 0):
                            continue # Ignore this hand for now

                        lm = raw_lms[idx]
                        if not lm: continue
                        
                        hand_dx = payloads[idx]["transform"]["delta"]["dx"] * w * 0.5
                        hand_dy = payloads[idx]["transform"]["delta"]["dy"] * h * 0.5
                        
                        for i, point in enumerate(lm.landmark):
                            px, py = point.x * w, point.y * h
                            
                            dist = ((px - box.x)**2 + (py - box.y)**2)**0.5
                            if dist < (box.size / 2 + 10):
                                rx = box.x - px
                                ry = box.y - py
                                
                                r_len = max(0.1, (rx**2 + ry**2)**0.5)
                                rx, ry = rx/r_len, ry/r_len
                                
                                push_power = 20.0
                                box.vx = (rx * push_power) + hand_dx
                                box.vy = (ry * push_power) - 10.0 + hand_dy
                                
                                if not box.hit:
                                    self.score += 1

                                box.hit = True
                                box.hit_cooldown = 4  # Drastically reduced hit cooldown for juggling
                                collision_found = True
                                break
                            
                box.draw(vis)
                
                # Despawn checking
                if box.y > h + box.size*2 or box.x < -box.size*2 or box.x > w + box.size*2:
                    box.active = False
                    
            self.boxes = [b for b in self.boxes if b.active]
            
            # Draw game state
            cv2.putText(vis, f"Score: {self.score}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(vis, "Make TRIANGLE to END", (w - 240, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw visual indicator for gravity hands
            for idx in valid_indices:
                side = payloads[idx]["header"]["hand_side"]
                if side in self.curr_fist_sides:
                    anchor = payloads[idx]["transform"]["anchor"]
                    ax, ay = int(anchor["x"] * w), int(anchor["y"] * h)
                    cv2.circle(vis, (ax, ay), 60, (0, 0, 255), 2)
                    cv2.putText(vis, "GRAVITY", (ax - 40, ay - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        else:
            # Idle text for Start
            msg = "Hold FIST with BOTH hands for 1.5s to TEST MODE"
            cv2.putText(vis, msg, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            if self.start_hold_time is not None:
                progress = int((timestamp - self.start_hold_time) / 1.5 * 100)
                cv2.putText(vis, f"Starting... {progress}%", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        return vis
