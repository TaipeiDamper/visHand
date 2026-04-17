# Changelog

本文件採用工程維運格式，記錄可追蹤的功能與相容性變更。

## [2026-04-15] Engine and Demo Update

### Added
- 啟用多手追蹤流程（`max_hands=2`），並確認 `demo.py` 命令列參數可正確載入左右手。
- 新增 Demo 沙盒模式：雙手握拳持續 1.5 秒進入，三角手勢持續 1 秒退出。
- 新增軌道重力互動：
  - 握拳手可作為重力場吸引場景物件。
  - 加入切向力與阻力，讓物件可在手部附近穩定公轉。
  - 重力手碰撞豁免與 0.2 秒釋放後免碰撞保護。
- 新增 Snap 衍生互動：
  - 重力手 Snap 觸發反向爆破。
  - 非重力手 Snap 觸發清場事件。

### Changed
- Snap 判定改為三訊號加權評分模型，閾值以 `snap_score_threshold` 控制。
- 握拳判定改為 `finger_curl` 浮點追蹤，並調整 `fist_curl_threshold` 至 `0.45`。

### Fixed
- 修正使用視窗 `X` 關閉時，背景迴圈未停止造成程序掛住的問題。

### Compatibility
- 輸出契約：無破壞性變更（既有 payload 消費端可持續運作）。
- 設定相容性：新增與調整皆為向後相容，未移除既有設定鍵。

### Breaking Changes
- 無。

## [2026-04-17] Grasp Pipeline Upgrades (Phase 1-4)

### Added
- `core/interpreter.py` 新增 Grasp FSM (`GRASP_OPEN/ENTERING/HOLD/DRAG`) 與 pinch 進出 hysteresis。
- `config/settings.py` 新增 grasp/association/calibration 相關設定欄位。
- `config/calibration_profile.py`：校正 profile 載入/套用與由 JSONL 估計 profile。
- `utils/math_tools.py` 新增 `pinch_contact_score()`；`GestureContext` 新增 `contact_score`。
- `core/inference.py` 新增 multi-hand slot association，穩定推論輸出順序。

### Changed
- `core/gestures.py` pinch evaluator 改用 `base_score * contact_score`。
- `bridge/INTEGRATION_CONTRACT.md` 增補 optional 欄位：`state.grasp_phase`, `state.contact_score`。
- `examples/validate_calibration.py` 支援 `--export-profile` 匯出 profile。
- `examples/gesture_calibration.py` 支援 `--export-profile`（按 E 匯出 session 後同時輸出 profile）。

### Compatibility
- Required payload 欄位維持不變，新增欄位皆為 optional。
- 未移除既有設定鍵，僅擴充新設定。

### Breaking Changes
- 無。

## [2026-04-17] Bridge Hardening for 3D Integrations (P0/P1)

### Added
- `bridge/transport.py` 實作可用 `WebSocketTransport`（內建 WS server）與 `NamedPipeTransport`（Windows pipe writer），並保留可配置 stdout fallback。
- `config/settings.py` 新增 bridge 連線設定與 feature flags：
  - `bridge_ws_host`, `bridge_ws_max_clients`, `bridge_ws_fallback_stdout`
  - `bridge_pipe_reconnect_ms`, `bridge_pipe_fallback_stdout`
  - `bridge_enable_extended_transform`, `bridge_enable_event_phase`, `bridge_enable_hand_identity`
- `core/interpreter.py` 可選輸出擴充欄位：
  - `header.dt_ms`
  - `transform.delta.dz`
  - `transform.rotation_euler.{roll,pitch,yaw}`
  - `dynamics.event_phase`
  - `header.hand_id`
- `core/inference.py` / `core/types.py` 新增 `HandLandmarkResult.track_id`，並在 hand association 產生穩定 `slot-*` identity。
- `bridge/schema_v1.py` 新增 `header.capabilities_meta`（`extended_transform`, `event_phase`, `hand_identity`）。

### Changed
- `bridge/INTEGRATION_CONTRACT.md` 擴充 optional 欄位、座標系約定與 3D payload 範例。
- `bridge/README.md` 補充 transport 行為、可靠性說明與新設定鍵。
- `demo.py` 在 payload 管線中傳遞 `hand_id`。

### Compatibility
- `schema_version` 維持 `1.0`。
- 原本 required 欄位不變；新增欄位皆為 optional 或由 flag 啟用。

### Breaking Changes
- 無。
