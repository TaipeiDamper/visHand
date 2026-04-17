from __future__ import annotations

from bridge.transport import BridgeTransport, NamedPipeTransport, StdoutTransport, WebSocketTransport


def create_transport(settings) -> BridgeTransport:
    mode = str(getattr(settings, "bridge_transport", "stdout")).strip().lower()
    if mode == "ws":
        return WebSocketTransport(
            port=getattr(settings, "bridge_ws_port", 9876),
            host=getattr(settings, "bridge_ws_host", "127.0.0.1"),
            fallback_to_stdout=getattr(settings, "bridge_ws_fallback_stdout", True),
            max_clients=getattr(settings, "bridge_ws_max_clients", 4),
        )
    if mode == "pipe":
        return NamedPipeTransport(
            pipe_name=getattr(settings, "bridge_pipe_name", "vishand"),
            fallback_to_stdout=getattr(settings, "bridge_pipe_fallback_stdout", True),
            reconnect_interval_ms=getattr(settings, "bridge_pipe_reconnect_ms", 1200),
        )
    return StdoutTransport()
