"""Same-thread wakeup eligibility. No simulator restart or action replay."""
import json

NETWORK_CONTINUE_DELAYS = (10,) * 20


def is_network_error(error):
    text = json.dumps(error, ensure_ascii=False).lower()
    # Account/config/safety failures must not become automatic model calls.
    if any(s in text for s in ('usage limit', 'quota', 'unauthorized', 'authentication',
                               'invalid api key', 'permission denied', 'safety',
                               'five rejected', 'exceeds 5 cm')):
        return False
    return any(s in text for s in ('stream disconnected', 'responsestreamdisconnected',
        'error decoding response body', 'network error', 'connection reset',
        'connection closed', 'connection timed out', 'tls close_notify'))


def closed_network_turn(response, thread_id, turn_id):
    """Timeout probe is read-only. Never wake an active or interrupted turn."""
    thread = (response or {}).get('thread', {})
    if thread.get('id') != thread_id or thread.get('status', {}).get('type') != 'idle':
        return None
    turns = thread.get('turns', [])
    if not turns or turns[-1].get('id') != turn_id:
        return None
    turn = turns[-1]
    return turn if turn.get('status') == 'failed' and is_network_error(turn.get('error')) else None
