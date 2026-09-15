"""Versioned, lossless RPC over an SSH-forwarded loopback TCP socket.

No pickle, automatic retries, or implicit reconnection of mutating requests.
An uncertain response invalidates the client; the next controller must reset.
"""
import socket
import struct
import uuid
import zlib

import msgpack
import numpy as np

VERSION = 1
MAX_BYTES = 128 * 1024 * 1024


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif":
            raise TypeError(f"Unsupported array dtype: {value.dtype}")
        value = np.ascontiguousarray(value)
        return msgpack.ExtType(42, msgpack.packb(
            (value.dtype.str, value.shape, value.tobytes()), use_bin_type=True))
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported RPC value: {type(value).__name__}")


def _decode(code, payload):
    if code != 42:
        raise ValueError("Unknown RPC extension")
    dtype, shape, data = msgpack.unpackb(payload, raw=False)
    dtype = np.dtype(dtype)
    if dtype.kind not in "buif" or len(shape) > 8 or any(x < 0 for x in shape):
        raise ValueError("Invalid RPC array")
    expected = int(np.prod(shape, dtype=object)) * dtype.itemsize
    if expected != len(data) or expected > MAX_BYTES:
        raise ValueError("RPC array size mismatch")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _read_exact(sock, length):
    result = bytearray()
    while len(result) < length:
        block = sock.recv(min(length - len(result), 1024 * 1024))
        if not block:
            raise EOFError("RPC connection closed")
        result.extend(block)
    return bytes(result)


def send_packet(sock, value):
    raw = msgpack.packb(value, default=_encode, use_bin_type=True)
    if len(raw) > MAX_BYTES:
        raise ValueError("RPC message too large")
    payload = zlib.compress(raw, level=1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Compressed RPC message too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_packet(sock):
    length, = struct.unpack("!I", _read_exact(sock, 4))
    if not 0 < length <= MAX_BYTES:
        raise ValueError("Invalid RPC packet length")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(_read_exact(sock, length), MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid compressed RPC packet")
    return msgpack.unpackb(raw, raw=False, ext_hook=_decode)


class RPCClient:
    def __init__(self, host, port, timeout=300):
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def request(self, op, **kwargs):
        if self.sock is None:
            raise RuntimeError("RPC client closed; create a client and reset")
        request_id = uuid.uuid4().hex
        try:
            send_packet(self.sock, dict(version=VERSION, request_id=request_id,
                                        op=op, args=kwargs))
            result = receive_packet(self.sock)
            if result.get("version") != VERSION or result.get("request_id") != request_id:
                raise ValueError("RPC response identity mismatch")
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "RPC request failed"))
            return result["result"]
        except Exception:
            self.close()
            raise

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
