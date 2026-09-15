"""Bounded OrcaRouter model discovery with capability filtering.

The single source of truth is ``GET <api_base>/v1/models`` on the configured
inference origin, optionally narrowed by ``?capability=``. Live results are
authoritative. When discovery fails, a small verified seed keeps a fresh
installation usable and is clearly reported as degraded; the seed is never mixed
into a successful live result.

Capabilities are read from catalog metadata only. A model's name is never used
to guess what it can do, and a model that does not declare a required modality
is excluded rather than admitted by default.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import origins

MAX_CATALOG_BYTES = 2_000_000
MAX_CATALOG_ITEMS = 5000
MAX_ID_LENGTH = 200
MAX_TEXT_FIELD = 4000
DISCOVERY_TIMEOUT = 20.0

#: Endpoint types this client can actually speak (Codex uses ``wire_api="responses"``).
TEXT_ENDPOINT_TYPES = ('openai', 'openai-response', 'anthropic', 'gemini')

#: Model families whose only purpose is media generation or reranking. They are
#: never valid text-chat targets even though the relay reports them under the
#: chat capability listing.
NON_TEXT_MARKERS = ('image-generation', 'openai-video', 'jina-rerank', 'embedding', 'rerank')

CAPABILITY_ENDPOINT_TYPES = {
    'chat': TEXT_ENDPOINT_TYPES,
    'embedding': ('embeddings',),
    'image': ('image-generation',),
    'video': ('openai-video',),
    'rerank': ('jina-rerank',),
}

SERVER_CAPABILITY_QUERY = {
    'chat': 'chat',
    'embedding': 'embedding',
    'image': 'image',
    'video': 'video',
    'rerank': 'rerank',
}

# Verified cold-start seed. Used only when live discovery fails, and always
# surfaced with ``degraded=True``. Metadata this release already verified is
# retained rather than flattened away, so enabling live discovery cannot quietly
# remove a reasoning ladder or an input modality.
VERIFIED_MODELS = (
    dict(id='openai/gpt-5.5', context_length=400000, input_modalities=('text', 'image'),
         reasoning_efforts=('low', 'medium', 'high', 'xhigh')),
    dict(id='anthropic/claude-opus-4.8', context_length=200000, input_modalities=('text', 'image')),
    dict(id='google/gemini-3.5-flash', context_length=1000000, input_modalities=('text', 'image')),
    dict(id='deepseek/deepseek-v4-pro', context_length=131072, input_modalities=('text',)),
    dict(id='orcarouter/auto', context_length=200000, input_modalities=('text',)),
)


class CatalogError(RuntimeError):
    """Discovery failed in a way the caller should report. Never carries a key."""


def verified_metadata(identifier):
    """Verified metadata recorded for a catalog id, or ``None``."""
    for entry in VERIFIED_MODELS:
        if entry['id'] == identifier:
            return entry
    return None


def seed_models():
    """Return the verified fallback seed as catalog records."""
    records = []
    for entry in VERIFIED_MODELS:
        model = dict(entry)
        model['input_modalities'] = list(model['input_modalities'])
        model['reasoning_efforts'] = list(model.get('reasoning_efforts', ()))
        model['endpoint_types'] = list(TEXT_ENDPOINT_TYPES)
        model['name'] = model['id']
        model['owned_by'] = None
        model['output_modalities'] = ['text']
        model['description'] = None
        records.append(model)
    return records


def _text(value, limit=MAX_TEXT_FIELD):
    return value[:limit] if isinstance(value, str) else None


def _string_list(value, limit=32):
    if not isinstance(value, list):
        return []
    return [item[:64] for item in value[:limit] if isinstance(item, str)]


def parse_model(raw):
    """Convert one catalog record into the shape the picker consumes.

    Returns ``None`` for anything that is not a usable record, so a malformed
    entry is dropped instead of reaching a selector.
    """
    if not isinstance(raw, dict):
        return None
    identifier = raw.get('id')
    # An over-long id is rejected rather than truncated: a truncated id would
    # name a model that does not exist and could still be selected.
    if not isinstance(identifier, str) or not identifier or len(identifier) > MAX_ID_LENGTH:
        return None
    if any(character.isspace() for character in identifier):
        return None
    architecture = raw.get('architecture') if isinstance(raw.get('architecture'), dict) else {}
    declared_modalities = _string_list(architecture.get('input_modalities'), 8)
    endpoint_types = _string_list(raw.get('supported_endpoint_types'))
    if not endpoint_types:
        endpoint_types = list(TEXT_ENDPOINT_TYPES)
    context = raw.get('context_length')
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        context = None
    model = dict(id=identifier, name=_text(raw.get('name'), 200) or identifier,
                 owned_by=_text(raw.get('owned_by'), 100),
                 context_length=context,
                 # Live metadata is authoritative when present. When the relay
                 # omits it, verified metadata for a known id fills the gap so
                 # discovery cannot erase a capability the release established.
                 input_modalities=declared_modalities or ['text'],
                 output_modalities=_string_list(architecture.get('output_modalities'), 8),
                 endpoint_types=endpoint_types,
                 reasoning_efforts=[],
                 description=_text(raw.get('description')))
    verified = verified_metadata(identifier)
    if verified:
        if not declared_modalities and verified.get('input_modalities'):
            model['input_modalities'] = list(verified['input_modalities'])
        if model['context_length'] is None and verified.get('context_length'):
            model['context_length'] = verified['context_length']
        if verified.get('reasoning_efforts'):
            model['reasoning_efforts'] = list(verified['reasoning_efforts'])
    return model


def _is_text_capable(model):
    """Chat requires a text endpoint type and must exclude media-only families."""
    if not any(kind in TEXT_ENDPOINT_TYPES for kind in model['endpoint_types']):
        return False
    if any(marker in model['endpoint_types'] for marker in NON_TEXT_MARKERS):
        return False
    outputs = model['output_modalities']
    if outputs and 'text' not in outputs:
        return False
    return True


def filter_models(models, capability='chat', required_modalities=()):
    """Apply one entry point's capability rules to a catalog.

    ``required_modalities`` must all be declared by ``architecture.input_modalities``.
    A model that does not declare them is dropped, so an undeclared capability
    fails closed instead of silently appearing in a multimodal picker.
    """
    filters = {
        'chat': lambda m: _is_text_capable(m),
        'embedding': lambda m: any(k in CAPABILITY_ENDPOINT_TYPES['embedding'] for k in m['endpoint_types']),
        'image': lambda m: any(k in CAPABILITY_ENDPOINT_TYPES['image'] for k in m['endpoint_types']),
        'video': lambda m: any(k in CAPABILITY_ENDPOINT_TYPES['video'] for k in m['endpoint_types']),
        'rerank': lambda m: any(k in CAPABILITY_ENDPOINT_TYPES['rerank'] for k in m['endpoint_types']),
    }
    if capability not in filters:
        raise CatalogError(f'Unknown capability: {capability}')
    wanted = [modality for modality in (required_modalities or ()) if modality != 'text']
    selected = []
    for model in models:
        if not filters[capability](model):
            continue
        if wanted and not all(modality in model['input_modalities'] for modality in wanted):
            continue
        selected.append(model)
    return selected


def is_compatible(model, capability='chat', required_modalities=()):
    """Whether an already-selected model still satisfies the current entry point."""
    return bool(filter_models([model], capability, required_modalities))


def parse_catalog(payload, capability='chat', required_modalities=()):
    """Parse and filter a raw ``/v1/models`` payload."""
    if not isinstance(payload, dict):
        raise CatalogError('Model catalog was not a JSON object.')
    data = payload.get('data')
    if not isinstance(data, list):
        raise CatalogError('Model catalog did not contain a model list.')
    models = []
    seen = set()
    for raw in data[:MAX_CATALOG_ITEMS]:
        model = parse_model(raw)
        if model is None or model['id'] in seen:
            continue
        seen.add(model['id'])
        models.append(model)
    return filter_models(models, capability, required_modalities)


def discovery_url(capability='chat', env=None):
    query = {'capability': SERVER_CAPABILITY_QUERY.get(capability, 'chat')} if capability else None
    return origins.url(origins.api_base(env), origins.MODELS_PATH, query)


def fetch_catalog(api_key, capability='chat', required_modalities=(),
                  timeout=DISCOVERY_TIMEOUT, env=None):
    """Fetch and filter the live catalog.

    Returns ``dict(models=[...], source='live', degraded=False, capability=...)``.
    A bounded request keeps a slow or hostile catalog response from consuming
    unbounded time or memory.
    """
    target = discovery_url(capability, env)
    request = urllib.request.Request(target, headers={
        'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_CATALOG_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise CatalogError('OrcaRouter rejected the stored key while listing models.') from None
        raise CatalogError(f'Model catalog request failed (HTTP {error.code}).') from None
    except (urllib.error.URLError, OSError, ValueError):
        raise CatalogError('Could not reach the OrcaRouter model catalog.') from None
    if len(raw) > MAX_CATALOG_BYTES:
        raise CatalogError('Model catalog response exceeded the size limit.')
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        raise CatalogError('Model catalog was not valid JSON.') from None
    models = parse_catalog(payload, capability, required_modalities)
    if not models:
        # An empty filtered list is a real answer, not an outage: the workspace
        # genuinely has no model of that kind. Do not paper over it with a seed.
        return dict(models=[], source='live', degraded=False, capability=capability,
                    required_modalities=list(required_modalities))
    return dict(models=models, source='live', degraded=False, capability=capability,
                required_modalities=list(required_modalities))


def catalog(api_key, capability='chat', required_modalities=(), timeout=DISCOVERY_TIMEOUT,
            env=None):
    """Live catalog with a bounded, clearly-labelled fallback when discovery fails."""
    try:
        return fetch_catalog(api_key, capability, required_modalities, timeout, env)
    except CatalogError as error:
        if capability != 'chat':
            # The seed only describes chat models; inventing embeddings or video
            # models for it would be a guess.
            return dict(models=[], source='unavailable', degraded=True, capability=capability,
                        required_modalities=list(required_modalities), reason=str(error))
        models = filter_models(seed_models(), 'chat', required_modalities)
        return dict(models=models, source='verified_seed', degraded=True, capability=capability,
                    required_modalities=list(required_modalities), reason=str(error))


def reconcile_selection(selected_id, models):
    """Keep a stored model only while it is still compatible with the entry point.

    Returns ``(model_id_or_None, changed)``. A selection that no longer appears
    in the filtered options is cleared rather than silently retained.
    """
    if not selected_id:
        return None, False
    if any(model['id'] == selected_id for model in models):
        return selected_id, False
    return None, True
