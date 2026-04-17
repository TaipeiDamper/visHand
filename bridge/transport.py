from __future__ import annotations

import base64
import hashlib
import json
import socket
import time
from typing import Protocol


class BridgeTransport(Protocol):
    def send_state(self, payload: dict) -> None:
        ...

    def send_event(self, event_name: str, payload: dict) -> None:
        ...

    def close(self) -> None:
        ...


class StdoutTransport:
    def send_state(self, payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False))

    def send_event(self, event_name: str, payload: dict) -> None:
        body = dict(payload)
        body["bridge_event"] = event_name
        print(json.dumps(body, ensure_ascii=False))

    def close(self) -> None:
        return


class WebSocketTransport:
    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        *,
        fallback_to_stdout: bool = True,
        max_clients: int = 4,
    ):
        self.host = str(host)
        self.port = int(port)
        self._stdout = StdoutTransport()
        self._fallback_to_stdout = bool(fallback_to_stdout)
        self._max_clients = max(1, int(max_clients))
        self._warned_no_client = False
        self._clients: list[socket.socket] = []
        self._server: socket.socket | None = None
        self._active = self._bind_server()

    def _bind_server(self) -> bool:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(self._max_clients)
            server.setblocking(False)
            self._server = server
            print(f"[visHand][bridge] WS transport listening at ws://{self.host}:{self.port}")
            return True
        except OSError as exc:
            print(f"[visHand][bridge] WS transport unavailable ({exc}).")
            self._server = None
            return False

    @staticmethod
    def _parse_headers(request_text: str) -> dict:
        headers = {}
        lines = request_text.split("\r\n")
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        return headers

    @staticmethod
    def _build_accept_key(client_key: str) -> str:
        magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        digest = hashlib.sha1((client_key + magic).encode("utf-8")).digest()
        return base64.b64encode(digest).decode("ascii")

    def _close_client(self, client: socket.socket) -> None:
        try:
            client.close()
        except OSError:
            pass
        self._clients = [c for c in self._clients if c is not client]

    def _accept_pending_clients(self) -> None:
        if self._server is None:
            return
        while True:
            try:
                client, _ = self._server.accept()
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                client.settimeout(0.5)
                request = client.recv(4096).decode("utf-8", errors="ignore")
                headers = self._parse_headers(request)
                ws_key = headers.get("sec-websocket-key")
                if not ws_key:
                    client.close()
                    continue
                accept_key = self._build_accept_key(ws_key)
                response = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
                )
                client.sendall(response.encode("utf-8"))
                client.setblocking(False)
                self._clients.append(client)
                self._warned_no_client = False
                print(f"[visHand][bridge] WS client connected ({len(self._clients)} active).")
            except OSError:
                try:
                    client.close()
                except OSError:
                    pass

    @staticmethod
    def _encode_ws_text_frame(text: str) -> bytes:
        payload = text.encode("utf-8")
        length = len(payload)
        if length <= 125:
            header = bytes([0x81, length])
        elif length <= 65535:
            header = bytes([0x81, 126]) + length.to_bytes(2, "big")
        else:
            header = bytes([0x81, 127]) + length.to_bytes(8, "big")
        return header + payload

    def _broadcast(self, payload: dict) -> bool:
        self._accept_pending_clients()
        if not self._clients:
            if self._fallback_to_stdout:
                self._stdout.send_state(payload)
            elif not self._warned_no_client:
                self._warned_no_client = True
                print("[visHand][bridge] WS has no connected client; dropping payload.")
            return False
        wire = self._encode_ws_text_frame(json.dumps(payload, ensure_ascii=False))
        dead: list[socket.socket] = []
        for client in self._clients:
            try:
                client.sendall(wire)
            except OSError:
                dead.append(client)
        for client in dead:
            self._close_client(client)
        return len(self._clients) > 0

    def send_state(self, payload: dict) -> None:
        if not self._active:
            if self._fallback_to_stdout:
                self._stdout.send_state(payload)
            return
        self._broadcast(payload)

    def send_event(self, event_name: str, payload: dict) -> None:
        body = dict(payload)
        body["bridge_event"] = event_name
        if not self._active:
            if self._fallback_to_stdout:
                self._stdout.send_event(event_name, payload)
            return
        self._broadcast(body)

    def close(self) -> None:
        for client in list(self._clients):
            self._close_client(client)
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None


class NamedPipeTransport:
    def __init__(
        self,
        pipe_name: str,
        *,
        fallback_to_stdout: bool = True,
        reconnect_interval_ms: int = 1200,
    ):
        self.pipe_name = str(pipe_name)
        self._pipe_path = rf"\\.\pipe\{self.pipe_name}"
        self._fallback_to_stdout = bool(fallback_to_stdout)
        self._reconnect_interval = max(0.1, float(reconnect_interval_ms) / 1000.0)
        self._next_retry_at = 0.0
        self._stdout = StdoutTransport()
        self._writer = None
        self._warned = False

    def _connect_if_needed(self) -> bool:
        if self._writer is not None:
            return True
        now = time.time()
        if now < self._next_retry_at:
            return False
        try:
            self._writer = open(self._pipe_path, "w", encoding="utf-8", buffering=1)
            self._warned = False
            print(f"[visHand][bridge] Named pipe connected: {self._pipe_path}")
            return True
        except OSError as exc:
            self._next_retry_at = now + self._reconnect_interval
            if not self._warned:
                print(f"[visHand][bridge] Named pipe unavailable ({exc}).")
                self._warned = True
            return False

    def _write_json(self, payload: dict, *, event_name: str | None = None) -> bool:
        if not self._connect_if_needed():
            if self._fallback_to_stdout:
                if event_name is None:
                    self._stdout.send_state(payload)
                else:
                    self._stdout.send_event(event_name, payload)
            return False
        try:
            assert self._writer is not None
            self._writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._writer.flush()
            return True
        except OSError:
            try:
                if self._writer is not None:
                    self._writer.close()
            except OSError:
                pass
            self._writer = None
            self._next_retry_at = time.time() + self._reconnect_interval
            if self._fallback_to_stdout:
                if event_name is None:
                    self._stdout.send_state(payload)
                else:
                    self._stdout.send_event(event_name, payload)
            return False

    def send_state(self, payload: dict) -> None:
        self._write_json(payload)

    def send_event(self, event_name: str, payload: dict) -> None:
        body = dict(payload)
        body["bridge_event"] = event_name
        self._write_json(body, event_name=event_name)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except OSError:
                pass
            self._writer = None
