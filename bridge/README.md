# visHand Bridge v1

`bridge` 目錄提供 visHand 對外橋樑層，目標是讓上游辨識邏輯與下游應用程式解耦。

## 文件責任

- 外部程式整合契約（唯一來源）：`bridge/INTEGRATION_CONTRACT.md`
- 本文件：Bridge 模組內部組成與程式進入點
- 系統總體資料流：`ARCHITECTURE_FLOW.md`

## 模組組成

- `schema_v1.py`
  - `ensure_v1_payload(payload, hand_side)`：補齊 v1 契約必要欄位
  - `capabilities_snapshot()`：輸出目前啟用的手勢能力清單
- `transport.py`
  - `BridgeTransport` protocol
  - `StdoutTransport` / `WebSocketTransport` / `NamedPipeTransport`
  - `WebSocketTransport`：內建簡易 WS server（text frame，JSON line payload）
  - `NamedPipeTransport`：Windows named pipe writer（`\\.\pipe\<name>`）
- `factory.py`
  - `create_transport(settings)`：依設定建立 transport instance

## 設定來源

- `config/settings.py`
  - `bridge_transport`: `"stdout" | "ws" | "pipe"`
  - `bridge_ws_host`
  - `bridge_ws_port`
  - `bridge_ws_max_clients`
  - `bridge_ws_fallback_stdout`
  - `bridge_pipe_name`
  - `bridge_pipe_reconnect_ms`
  - `bridge_pipe_fallback_stdout`
  - `bridge_enable_extended_transform`
  - `bridge_enable_event_phase`
  - `bridge_enable_hand_identity`

## Reliability Notes

- `ws` / `pipe` 模式都支援 fallback 到 stdout（可由設定關閉）。
- `ws` 模式會在啟動時監聽，無 client 時可選擇丟棄或 fallback。
- `pipe` 模式採重試連線，避免 consumer 晚於 visHand 啟動時直接失效。

## 最小使用範例
```python
from bridge.factory import create_transport

transport = create_transport(settings)
transport.send_state(payload)
transport.send_event("SNAP", payload)
transport.close()
```
