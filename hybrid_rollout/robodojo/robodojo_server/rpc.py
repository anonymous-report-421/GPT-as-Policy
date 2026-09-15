"""Single-owner blocking RPC; disconnects finalize evidence, never retry actions."""
import json
import socket
import traceback
from .protocol import VERSION, receive_packet, send_packet


def serve(session, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
        listener.listen(1)
        print(json.dumps(dict(event='ready', port=port, metadata=session.metadata)), flush=True)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(900)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while True:
                    request = receive_packet(connection)
                    response = dict(version=VERSION, request_id=request.get('request_id'))
                    try:
                        if request.get('version') != VERSION:
                            raise ValueError('Unsupported protocol version')
                        result = session.dispatch(request['op'], request.get('args', {}))
                        response.update(ok=True, result=result)
                    except Exception as exc:
                        session.poisoned = True
                        traceback.print_exc()
                        response.update(ok=False, error=f'{type(exc).__name__}: {exc}')
                    send_packet(connection, response)
            except (EOFError, ConnectionError, TimeoutError):
                session._write_summary('terminal' if session.terminated or session.truncated else 'controller_disconnect')
