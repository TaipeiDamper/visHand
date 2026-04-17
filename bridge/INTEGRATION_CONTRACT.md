# visHand Integration Contract (Bridge v1)

這份文件是給「外部程式整合者」的最小契約。  
目標：降低嫁接成本，避免下游依賴 visHand 內部實作細節。

## 1) Contract Scope

- 協議版本：`header.schema_version == "1.0"`
- 輸出型態：
  - `state stream`：每幀狀態
  - `event stream`：離散事件（例如 `SNAP`、`DOUBLE_TAP`）
- 傳輸層與手勢引擎解耦：外部整合只需吃 payload JSON

## 2) Required Fields (Must Read)

外部程式至少應讀取以下欄位：

- `header.schema_version`
- `header.frame_id`
- `header.timestamp`
- `header.hand_side`
- `state.logic`
- `state.intent`
- `state.intent_confidence`
- `state.input_quality_score`
- `state.tracking_quality`
- `transform.anchor.{x,y,z}`
- `transform.delta.{dx,dy}`
- `dynamics.event`
- `dynamics.emergency_cancel`

## 2.1) Optional Fields (Forward-Compatible)

若存在可使用，若不存在請視為舊版 payload：

- `state.grasp_phase` (`GRASP_OPEN` / `GRASP_ENTERING` / `GRASP_HOLD` / `GRASP_DRAG`)
- `state.contact_score` (0.0 ~ 1.0)
- `header.dt_ms`（幀間隔，毫秒）
- `header.hand_id`（穩定手部識別，例如 `slot-0`）
- `header.coord_space`、`header.axis_convention`、`header.unit_scale`
- `transform.delta.dz`
- `transform.rotation_euler.{roll,pitch,yaw}`
- `dynamics.event_phase` (`NONE` / `START` / `UPDATE` / `END`)
- `header.capabilities_meta.{extended_transform,event_phase,hand_identity}`

## 2.2) Coordinate Convention (When Provided)

若 payload 內存在 `header.coord_space`，目前約定如下：

- `coord_space = "camera_normalized_right_handed"`
- `unit_scale = "normalized_0_1_xy"`
- `axis_convention`：
  - `x`: `right_positive`
  - `y`: `down_positive`
  - `z`: `away_from_camera_positive`

建議整合端在初始化階段做一次座標轉換設定，避免 Unity / Unreal / WebGL 軸向不一致。

## 3) Compatibility Rules

- **版本檢查**：若 `schema_version != "1.0"`，建議拒收或進入 fallback parser。
- **前向相容**：允許出現未知欄位，外部程式應忽略而不是報錯。
- **欄位缺失**：若非必要欄位缺失，使用預設值；必要欄位缺失時丟棄該訊息。
- **事件處理**：`dynamics.event == "NONE"` 視為無離散事件。
- **事件相位**：若 `dynamics.event_phase` 存在，`START/END` 視為 edge-trigger，`UPDATE` 視為持續互動幀。
- **安全優先**：`dynamics.emergency_cancel == true` 時，外部程式應立即停止高風險動作。

## 4) Minimal Consumer Logic

建議下游以這個順序處理：

1. 檢查 `header.schema_version`
2. 檢查 `dynamics.emergency_cancel`
3. 根據 `state.tracking_quality` 決定是否降級操作
4. 使用 `state.intent` + `transform` 做控制映射
5. 使用 `dynamics.event` 處理一次性觸發

## 5) Example Payload (Abbreviated)

```json
{
  "header": {
    "schema_version": "1.0",
    "frame_id": 1284,
    "timestamp": 1765432100.153,
    "dt_ms": 33.31,
    "hand_side": "RIGHT",
    "hand_id": "slot-0",
    "coord_space": "camera_normalized_right_handed",
    "axis_convention": {
      "x": "right_positive",
      "y": "down_positive",
      "z": "away_from_camera_positive"
    },
    "unit_scale": "normalized_0_1_xy",
    "capabilities": ["OPEN_PALM", "PINCH_HOLD", "SNAP"],
    "capabilities_meta": {
      "extended_transform": true,
      "event_phase": true,
      "hand_identity": true
    }
  },
  "state": {
    "logic": "ACTIVE",
    "intent": "PINCH_DRAG",
    "intent_confidence": 0.84,
    "tracking_quality": "good",
    "input_quality_score": 0.91
  },
  "transform": {
    "anchor": {"x": 0.533, "y": 0.418, "z": -0.042},
    "delta": {"dx": 0.0042, "dy": -0.0011, "dz": 0.0007},
    "rotation": 13.9,
    "rotation_euler": {"roll": 13.9, "pitch": -2.1, "yaw": 4.8}
  },
  "dynamics": {
    "event": "NONE",
    "event_phase": "UPDATE",
    "emergency_cancel": false
  }
}
```

## 6) Stability Promise (Bridge v1)

在 `1.x` 期間：

- 不移除上述 Required Fields
- 新增欄位只做擴充，不破壞既有解析
- 若有破壞性修改，會升到 `2.0`
