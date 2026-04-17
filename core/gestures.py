"""
visHand — Gesture Definitions
===============================
集中管理所有手勢定義，分為三大區塊：

  ① MEDIAPIPE_GESTURES   — MediaPipe GestureRecognizer 內建的 7 種手勢
  ② CUSTOM_GESTURES      — 使用者自訂手勢（基於 Landmark 關係自行定義）
  ③ COMPOSITE_GESTURES   — 複合動作（結合多個單一動作、時序或特殊節點條件）

每個手勢以 GestureDef dataclass 描述。新的架構支援動態注入 `evaluator`，
使手勢判斷不再是生硬的 if-else，而是透過信心指數 (Confidence) 來抉擇。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from core.context import GestureContext
import utils.math_tools as mt


# ═══════════════════════════════════════════════════════════════════════════
#  Data structure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GestureDef:
    """單一手勢的定義描述。"""
    name: str                       
    display_name: str               
    category: str                   
    description: str                
    suggested_usage: str            
    enabled: bool = True            
    detection_hint: str = ""        
    
    # 優先權：數字越大越優先判斷 (決定手勢的競爭順位)
    priority: int = 100

    # 狀態評估器：給定 GestureContext，回傳信心指數 (0.0 ~ 1.0)
    evaluator: Optional[Callable[[GestureContext], float]] = None
    risk_level: str = "medium"        # "low" | "medium" | "high"
    mutex_group: str = "pose"         # 同組手勢只允許一個勝出
    cooldown_ms: float = 0.0
    arm_assist_allowed: bool = False
    arm_assist_weight: float = 0.0

    # ── 複合動作專用欄位 ──────────────────────────────────────────────────
    sub_gestures: List[str] = field(default_factory=list)
    composite_type: str = ""
    composite_params: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluators (狀態評估器 - 回傳 0.0~1.0)
# ═══════════════════════════════════════════════════════════════════════════

def _eval_fist(ctx: GestureContext) -> float:
    # 拳頭：食指到小指皆不伸直，且彎曲度大於門檻
    curls = ctx.curl_array[1:] 
    if all(not ctx.is_extended(i) for i in range(1, 5)) and all(c > ctx.settings.fist_curl_threshold for c in curls):
        avg_curl = sum(curls) / 4.0
        return min(avg_curl / 0.8, 1.0) # 越捲信心度越高
    return 0.0

def _eval_open_palm(ctx: GestureContext) -> float:
    # 掌心全部張開
    if all(ctx.ext_array):
        return 1.0
    return 0.0

def _eval_pointing_up(ctx: GestureContext) -> float:
    ext = ctx.ext_array
    if ext[1] and not ext[2] and not ext[3] and not ext[4]:
        return 1.0
    return 0.0

def _eval_victory(ctx: GestureContext) -> float:
    ext = ctx.ext_array
    if ext[1] and ext[2] and not ext[3] and not ext[4]:
        return 1.0
    return 0.0

def _eval_thumb_up(ctx: GestureContext) -> float:
    ext = ctx.ext_array
    if ext[0] and not ext[1] and not ext[2] and not ext[3] and not ext[4]:
        # 拇指指尖 Y 需小於 拇指關節 Y (畫面中朝上)
        if ctx.lm[4].y < ctx.lm[2].y:
            return 1.0
    return 0.0

def _eval_thumb_down(ctx: GestureContext) -> float:
    ext = ctx.ext_array
    if ext[0] and not ext[1] and not ext[2] and not ext[3] and not ext[4]:
        # 拇指指尖 Y 需大於 拇指關節 Y (畫面中朝下)
        if ctx.lm[4].y > ctx.lm[2].y:
            return 1.0
    return 0.0

def _eval_iloveyou(ctx: GestureContext) -> float:
    ext = ctx.ext_array
    if ext[0] and ext[1] and not ext[2] and not ext[3] and ext[4]:
        return 1.0
    return 0.0

def _eval_pinch_hold(ctx: GestureContext) -> float:
    dist = ctx.pinch_distance
    if dist < ctx.settings.pinch_threshold:
        if ctx.velocity <= ctx.settings.drag_start_velocity:
            base = 1.0 - (dist / ctx.settings.pinch_threshold) # 捏越緊分數越高
            return max(0.0, min(1.0, base * ctx.contact_score))
    return 0.0

def _eval_pinch_drag(ctx: GestureContext) -> float:
    dist = ctx.pinch_distance
    if dist < ctx.settings.pinch_threshold:
        if ctx.velocity > ctx.settings.drag_start_velocity:
            base = 1.0 - (dist / ctx.settings.pinch_threshold)
            return max(0.0, min(1.0, base * ctx.contact_score))
    return 0.0

def _eval_snap_ready(ctx: GestureContext) -> float:
    dist = ctx.snap_ready_distance
    if dist < ctx.settings.snap_ready_threshold:
        return 1.0
    return 0.0


def _eval_ok_sign(ctx: GestureContext) -> float:
    dist = ctx.pinch_distance
    ext = ctx.ext_array
    if dist < ctx.settings.ok_sign_threshold and ext[2] and ext[3] and ext[4]:
        return max(0.0, 1.0 - (dist / max(ctx.settings.ok_sign_threshold, 1e-6)))
    return 0.0


def _eval_index_point_left(ctx: GestureContext) -> float:
    ext = ctx.ext_array
    if ext[1] and not ext[2] and not ext[3] and not ext[4]:
        dx = ctx.lm[mt.INDEX_TIP].x - ctx.lm[mt.INDEX_MCP].x
        if dx < -ctx.settings.index_point_lateral_threshold:
            return min(1.0, abs(dx) / max(ctx.settings.index_point_lateral_threshold * 2.0, 1e-6))
    return 0.0


def _eval_index_point_right(ctx: GestureContext) -> float:
    ext = ctx.ext_array
    if ext[1] and not ext[2] and not ext[3] and not ext[4]:
        dx = ctx.lm[mt.INDEX_TIP].x - ctx.lm[mt.INDEX_MCP].x
        if dx > ctx.settings.index_point_lateral_threshold:
            return min(1.0, abs(dx) / max(ctx.settings.index_point_lateral_threshold * 2.0, 1e-6))
    return 0.0


def _eval_palm_tilt_left(ctx: GestureContext) -> float:
    angle = mt.palm_roll_angle(ctx.lm)
    th = ctx.settings.palm_tilt_angle_threshold_deg
    if angle < -th:
        return min(1.0, abs(angle) / max(th * 2.0, 1e-6))
    return 0.0


def _eval_palm_tilt_right(ctx: GestureContext) -> float:
    angle = mt.palm_roll_angle(ctx.lm)
    th = ctx.settings.palm_tilt_angle_threshold_deg
    if angle > th:
        return min(1.0, abs(angle) / max(th * 2.0, 1e-6))
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  ① MediaPipe 原始手勢定義
# ═══════════════════════════════════════════════════════════════════════════

MEDIAPIPE_GESTURES: List[GestureDef] = [
    GestureDef(
        name="CLOSED_FIST",
        display_name="握拳",
        category="mediapipe",
        description="五指完全握緊成拳頭",
        suggested_usage="切換模式 / 停止",
        priority=50,
        evaluator=_eval_fist,
        risk_level="low",
        mutex_group="pose_base",
    ),
    GestureDef(
        name="OPEN_PALM",
        display_name="手掌張開",
        category="mediapipe",
        description="五指完全展開、手掌攤平",
        suggested_usage="重置 / 釋放",
        priority=50,
        evaluator=_eval_open_palm,
        risk_level="low",
        mutex_group="pose_base",
    ),
    GestureDef(
        name="POINTING_UP",
        display_name="食指向上",
        category="mediapipe",
        description="僅食指向上伸直",
        suggested_usage="導航 / Hover",
        priority=60,
        evaluator=_eval_pointing_up,
        risk_level="medium",
        mutex_group="pointing",
        arm_assist_allowed=True,
        arm_assist_weight=0.12,
    ),
    GestureDef(
        name="THUMB_UP",
        display_name="大拇指朝上",
        category="mediapipe",
        description="大拇指朝上，其餘手指握拳",
        suggested_usage="確認 / 增加",
        priority=60,
        evaluator=_eval_thumb_up,
        risk_level="low",
        mutex_group="pose_base",
    ),
    GestureDef(
        name="THUMB_DOWN",
        display_name="大拇指朝下",
        category="mediapipe",
        description="大拇指朝下，其餘手指握拳",
        suggested_usage="取消 / 減少",
        priority=60,
        evaluator=_eval_thumb_down,
        risk_level="low",
        mutex_group="pose_base",
    ),
    GestureDef(
        name="VICTORY",
        display_name="YA / 剪刀",
        category="mediapipe",
        description="食指與中指 V 字張開",
        suggested_usage="二選一切換",
        priority=70,
        evaluator=_eval_victory,
        risk_level="low",
        mutex_group="pose_base",
    ),
    GestureDef(
        name="I_LOVE_YOU",
        display_name="搖滾 / 我愛你",
        category="mediapipe",
        description="大拇指、食指、小指伸直",
        suggested_usage="自定義魔法動作",
        priority=70,
        evaluator=_eval_iloveyou,
        risk_level="medium",
        mutex_group="pose_base",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
#  ② 自訂手勢定義（Custom Gestures）
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_GESTURES: List[GestureDef] = [
    GestureDef(
        name="PINCH_HOLD",
        display_name="捏住",
        category="custom",
        description="拇指尖與食指尖靠近（靜止狀態）",
        suggested_usage="精確控制的起點",
        priority=90,  # 捏合判斷優先順序高於指向
        evaluator=_eval_pinch_hold,
        risk_level="medium",
        mutex_group="pinch_family",
    ),
    GestureDef(
        name="PINCH_DRAG",
        display_name="捏住拖曳",
        category="custom",
        description="拇指尖與食指尖捏住並移動",
        suggested_usage="拖曳物件",
        priority=100, # 動態操作最優先判定
        evaluator=_eval_pinch_drag,
        risk_level="medium",
        mutex_group="pinch_family",
    ),
    GestureDef(
        name="SNAP_READY",
        display_name="待彈指",
        category="custom",
        description="中指與拇指靠在一起",
        suggested_usage="預告即將觸發的動作",
        priority=110, # 最優先視覺反饋
        evaluator=_eval_snap_ready,
        risk_level="high",
        mutex_group="event_ready",
        cooldown_ms=80.0,
    ),
    GestureDef(
        name="OK_SIGN",
        display_name="OK",
        category="custom",
        description="拇指與食指形成圈，其他手指伸展",
        suggested_usage="確定 / 接受",
        priority=88,
        evaluator=_eval_ok_sign,
        risk_level="low",
        mutex_group="pinch_family",
    ),
    GestureDef(
        name="INDEX_POINT_LEFT",
        display_name="食指向左",
        category="custom",
        description="食指伸直且朝左指向",
        suggested_usage="左向導航",
        priority=82,
        evaluator=_eval_index_point_left,
        risk_level="low",
        mutex_group="pointing",
        arm_assist_allowed=True,
        arm_assist_weight=0.22,
    ),
    GestureDef(
        name="INDEX_POINT_RIGHT",
        display_name="食指向右",
        category="custom",
        description="食指伸直且朝右指向",
        suggested_usage="右向導航",
        priority=82,
        evaluator=_eval_index_point_right,
        risk_level="low",
        mutex_group="pointing",
        arm_assist_allowed=True,
        arm_assist_weight=0.22,
    ),
    GestureDef(
        name="PALM_TILT_LEFT",
        display_name="手掌左傾",
        category="custom",
        description="手掌roll角度朝左側偏轉",
        suggested_usage="微調旋轉",
        priority=78,
        evaluator=_eval_palm_tilt_left,
        risk_level="low",
        mutex_group="tilt",
    ),
    GestureDef(
        name="PALM_TILT_RIGHT",
        display_name="手掌右傾",
        category="custom",
        description="手掌roll角度朝右側偏轉",
        suggested_usage="微調旋轉",
        priority=78,
        evaluator=_eval_palm_tilt_right,
        risk_level="low",
        mutex_group="tilt",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
#  ③ 複合動作定義（Composite Gestures）
# ═══════════════════════════════════════════════════════════════════════════

COMPOSITE_GESTURES: List[GestureDef] = [
    GestureDef(
        name="SNAP",
        display_name="彈指",
        category="composite",
        description="中指與拇指接觸後快速彈開",
        suggested_usage="觸發一次性事件",
        sub_gestures=["SNAP_READY"],
        composite_type="sequence",
        composite_params={"cooldown_ms": 100},
        risk_level="high",
        mutex_group="event",
        cooldown_ms=100.0,
    ),
    GestureDef(
        name="DOUBLE_TAP",
        display_name="雙擊捏",
        category="composite",
        description="短時間內捏兩次",
        suggested_usage="雙擊確認",
        sub_gestures=["PINCH_HOLD", "PINCH_HOLD"],
        composite_type="sequence",
        composite_params={"timeout_ms": 350},
        risk_level="high",
        mutex_group="event",
        cooldown_ms=250.0,
    ),
    GestureDef(
        name="SWIPE_LEFT",
        display_name="左揮",
        category="composite",
        description="手掌張開並快速向左移動",
        suggested_usage="向左翻頁",
        sub_gestures=["OPEN_PALM"],
        composite_type="velocity",
        composite_params={"velocity_min": 0.04, "direction": "left"},
        risk_level="high",
        mutex_group="event",
        cooldown_ms=180.0,
        arm_assist_allowed=True,
        arm_assist_weight=0.18,
    ),
    GestureDef(
        name="SWIPE_RIGHT",
        display_name="右揮",
        category="composite",
        description="手掌張開並快速向右移動",
        suggested_usage="向右翻頁",
        sub_gestures=["OPEN_PALM"],
        composite_type="velocity",
        composite_params={"velocity_min": 0.04, "direction": "right"},
        risk_level="high",
        mutex_group="event",
        cooldown_ms=180.0,
        arm_assist_allowed=True,
        arm_assist_weight=0.18,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════

class GestureRegistry:
    def __init__(self):
        self._gestures: Dict[str, GestureDef] = {}

    def register(self, gesture: GestureDef) -> None:
        self._gestures[gesture.name] = gesture

    def register_many(self, gestures: List[GestureDef]) -> None:
        for g in gestures:
            self.register(g)

    def get(self, name: str) -> Optional[GestureDef]:
        return self._gestures.get(name)

    def all(self) -> List[GestureDef]:
        return list(self._gestures.values())

    def enabled_intents(self) -> List[GestureDef]:
        # 回傳所有啟用且具有 Evaluator 的靜態手勢 (Intent)，按優先權排序
        intents = [g for g in self._gestures.values() if g.enabled and g.evaluator is not None]
        intents.sort(key=lambda g: g.priority, reverse=True)
        return intents

    def event_defs(self) -> List[GestureDef]:
        events = [g for g in self._gestures.values() if g.category == "composite" and g.enabled]
        return events

    def set_enabled(self, name: str, enabled: bool) -> None:
        g = self._gestures.get(name)
        if g: g.enabled = enabled

    def by_category(self, category: str) -> List[GestureDef]:
        return [g for g in self._gestures.values() if g.category == category]

    def summary(self) -> str:
        lines = []
        for cat_label, cat_key in [("MediaPipe", "mediapipe"), ("Custom", "custom"), ("Composite", "composite")]:
            group = self.by_category(cat_key)
            if not group: continue
            lines.append(f"\n{cat_label} ({len(group)})")
            for g in group:
                lines.append(f"  [{'v' if g.enabled else 'x'}] {g.name} (Priority: {g.priority})")
        return "\n".join(lines)


registry = GestureRegistry()
registry.register_many(MEDIAPIPE_GESTURES)
registry.register_many(CUSTOM_GESTURES)
registry.register_many(COMPOSITE_GESTURES)
