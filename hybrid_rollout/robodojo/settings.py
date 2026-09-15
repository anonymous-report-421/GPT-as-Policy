"""Pinned rollout-only gateway; historical source snapshots retain their identity."""

from .codex_backend.profiles import profile

AUTH_PROFILE = profile()['name']
AUTH_MODE = profile()['auth_mode']
PROVIDER = profile()['provider']
MODEL = 'gpt-6-astra'
EFFORT = 'xhigh'
BASE_URL = profile()['base_url']
WIRE_API = 'responses'
DEFAULT_CODEX_IMAGE_MAX_EDGE = 480  # teacher attachments only; 0 keeps originals
