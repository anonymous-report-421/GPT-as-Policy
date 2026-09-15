"""Real-browser evidence for the OrcaRouter provider console.

Launches the actual loopback console against the live OrcaRouter catalog and
drives it with Chromium to produce ``orca-evidence/``: screenshots plus a
manifest recording what was asserted and how many models the catalog returned.

Skipped unless a Playwright runtime and a Chromium binary are configured, so a
plain ``pytest`` run on a machine without them stays green. No credential value
ever reaches the browser: the key is stored server side and only its masked form
is rendered.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from .codex_backend.orcarouter import console, provider
from .codex_backend.orcarouter.credentials import CredentialStore
from .codex_backend.profiles import private_dir

EVIDENCE = Path(__file__).resolve().parents[2]/'orca-evidence'
CATALOG_SOURCE = 'https://api.orcarouter.ai/v1/models?capability=chat'
VIEWPORT = {'width': 1280, 'height': 900}
MIN_WIDTH, MIN_HEIGHT = 800, 450


def _playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover - optional local runtime
        return None
    return sync_playwright


def _chromium_path():
    configured = os.environ.get('ORCA_CHROMIUM')
    if configured and Path(configured).exists():
        return configured
    return shutil.which('chromium') or shutil.which('chromium-browser')


requires_browser = pytest.mark.skipif(
    _playwright() is None or not _chromium_path(),
    reason='Optional local Playwright/Chromium runtime not configured')


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LiveConsole:
    """The real console server backed by a temporary private store.

    The credential rendered in the console is a synthetic fixture, so no shape of
    a real key can end up in a committed screenshot or manifest. The catalog is
    still fetched live: ``discovery_credential`` supplies the real key for the
    server-side discovery request only, and it is never displayed.
    """

    DISPLAYED_KEY = 'sk-orca-' + 'Ev1dence0nly' + 'x' * 20

    def __init__(self, tmp_path, api_key=None):
        self.directory = tmp_path/'private'
        private_dir(self.directory)
        self.store = CredentialStore(directory=self.directory)
        # Exercise the real API-key adapter, server side only.
        provider.adapter_for(provider.API_PROVIDER, self.store).acquire(self.DISPLAYED_KEY)
        from .codex_backend.orcarouter.credentials import Credential
        self.discovery = Credential(api_key, 'api_key') if api_key else None
        self.server = console.serve(self.store, '127.0.0.1', 0,
                                    discovery_credential=self.discovery)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': 0.1}, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/'

    def close(self):
        self.server.orca_manager.cancel()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        # The temporary store is discarded with tmp_path; ensure the file is gone.
        for path in self.directory.glob('*'):
            try:
                path.unlink()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@requires_browser
def test_pagehide_clears_busy_state_and_a_second_login_starts_without_remount(tmp_path):
    """The back-forward-cache shape, driven in a real browser.

    Browsers may restore a page from the back-forward cache without remounting
    it, so the invalidated request's guarded ``finally`` never runs. The
    ``pagehide`` handler must therefore clear the busy flag and the hint itself,
    and a second login must be startable on the same page object.
    """
    api_key = os.environ.get('ORCAROUTER_API_KEY')
    if not api_key:
        pytest.skip('ORCAROUTER_API_KEY not configured')
    sync_playwright = _playwright()

    with LiveConsole(tmp_path, api_key) as live, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=_chromium_path(),
                                             args=['--no-sandbox'])
        try:
            page = browser.new_page(viewport=VIEWPORT)
            errors = []
            page.on('pageerror', lambda error: errors.append(error.message))
            page.goto(live.url)
            page.wait_for_function(
                "() => document.getElementById('apiState').textContent !== 'checking'")

            # Begin a real authorization attempt: the console starts its loopback
            # listener and returns the consent URL. No consent is simulated.
            page.locator('#connect').click()
            page.wait_for_function("() => window.__orca.state().session !== null")
            first = page.evaluate('window.__orca.state()')
            assert first['busy'] is True, 'the login lock must be held while pending'
            assert page.locator('#connect').is_disabled()
            assert page.locator('#authUrl').text_content().startswith(
                'https://www.orcarouter.ai/auth?')
            assert page.locator('#oauthHint').text_content()

            # Dispatch pagehide exactly as a bfcache navigation would.
            page.evaluate("() => window.dispatchEvent(new PageTransitionEvent('pagehide'))")
            page.wait_for_function("() => window.__orca.state().busy === false")
            after = page.evaluate('window.__orca.state()')
            assert after['busy'] is False, 'busy must clear synchronously on pagehide'
            assert after['session'] is None, 'the attempt must be released'
            assert page.locator('#connect').is_enabled(), 'Connect must be usable again'
            assert page.locator('#cancel').is_disabled()
            assert page.locator('#oauthHint').text_content() == '', \
                'the authorization hint must clear'
            assert page.locator('#authUrl').text_content() == ''

            # The server-side lock is released too, so a second login can start.
            page.wait_for_function(
                """async () => {
                    const r = await fetch('/api/connect/poll', {cache:'no-store'});
                    return (await r.json()).state === 'none';
                }""")
            page.locator('#connect').click()
            page.wait_for_function("() => window.__orca.state().session !== null")
            second = page.evaluate('window.__orca.state()')
            assert second['session'] != first['session'], 'a new attempt must be minted'
            assert second['generation'] > first['generation'], 'the generation must advance'
            assert second['busy'] is True
            # The stale attempt cannot overwrite the newer one.
            page.evaluate("() => window.__orca.state()")

            # Leave no listener behind.
            page.locator('#cancel').click()
            page.wait_for_function("() => window.__orca.state().session === null")
            assert errors == [], errors
        finally:
            browser.close()


@requires_browser
def test_console_produces_gui_evidence_with_the_live_catalog(tmp_path):
    api_key = os.environ.get('ORCAROUTER_API_KEY')
    if not api_key:
        pytest.skip('ORCAROUTER_API_KEY not configured')
    sync_playwright = _playwright()
    EVIDENCE.mkdir(exist_ok=True)
    # Each artifact carries its own assertions, so a reviewer (or an independent
    # validator) reads the claim next to the image it belongs to.
    artifacts, ui = {}, {}
    text_count = image_count = 0
    errors = []

    with LiveConsole(tmp_path, api_key) as live, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=_chromium_path(),
                                             args=['--no-sandbox'])
        try:
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
            page.on('pageerror', lambda error: errors.append(error.message))
            page.goto(live.url)
            page.wait_for_selector('#modelTrigger')

            # ---- auth-methods.png: both choices visible and usable ----------
            page.wait_for_function(
                "() => document.getElementById('apiState').textContent !== 'checking'")
            # Wait for the live catalog too, so the screenshot shows the loaded state.
            page.wait_for_function(
                "() => /live catalog/.test(document.getElementById('catalogMeta').textContent)",
                timeout=60000)
            api_state = page.locator('#apiState').text_content()
            oauth_state = page.locator('#oauthState').text_content()
            masked = page.locator('#apiMasked').text_content()
            auth_ui = ui['auth-methods'] = {}
            auth_ui['api_key_visible'] = page.locator('#apiKey').is_visible()
            auth_ui['pkce_visible'] = page.locator('#connect').is_visible()
            auth_ui['controls_enabled'] = (page.locator('#saveKey').is_enabled()
                                           and page.locator('#connect').is_enabled()
                                           and page.locator('#clearKey').is_enabled())
            auth_ui['secret_masked'] = (masked.startswith('sk-orca-') and '…' in masked)
            auth_ui['api_key_label'] = page.locator('#apiKeyHeading').text_content()
            auth_ui['pkce_label'] = page.locator('#oauthHeading').text_content()
            auth_ui['masked_text'] = masked
            assert auth_ui['api_key_visible'] and auth_ui['pkce_visible']
            assert auth_ui['controls_enabled'], 'both authentication controls must be usable'
            assert api_state == 'connected' and oauth_state == 'not configured'
            assert auth_ui['secret_masked'], 'the stored key must be shown masked'
            # Neither the real credential nor the synthetic one is rendered whole.
            assert api_key not in masked, 'the raw key must never be rendered'
            assert LiveConsole.DISPLAYED_KEY not in masked
            auth_ui['api_key_label_distinct'] = (auth_ui['api_key_label']
                                                 != auth_ui['pkce_label'])
            page.screenshot(path=EVIDENCE/'auth-methods.png')
            artifacts['auth-methods'] = dict(kind='auth-methods', path='auth-methods.png',
                                             sha256=_sha256(EVIDENCE/'auth-methods.png'),
                                             ui=auth_ui)

            # ---- text-model-dropdown.png: real, expanded catalog ------------
            meta = page.locator('#catalogMeta').text_content()
            text_count = int(page.evaluate('window.__orca.state().models'))
            assert text_count > 0, meta
            page.locator('#modelTrigger').click()
            page.wait_for_selector('#modelPanel .opt')
            panel = page.locator('#modelPanel')
            trigger = page.locator('#modelTrigger')
            text_ui = ui['text-model-dropdown'] = {}
            text_ui['dropdown_open'] = panel.is_visible()
            text_ui['item_count'] = panel.locator('.opt').count()
            box_trigger = trigger.bounding_box()
            box_panel = panel.bounding_box()
            text_ui['trigger_panel_right_delta'] = round(
                abs((box_trigger['x'] + box_trigger['width'])
                    - (box_panel['x'] + box_panel['width'])), 2)
            style = panel.evaluate("el => getComputedStyle(el)")
            text_ui['opaque_background'] = ('rgba(0, 0, 0, 0)' != style['backgroundColor'])
            text_ui['visible_border'] = (style['borderTopStyle'] != 'none'
                                         and float(style['borderTopWidth'].rstrip('px')) > 0)
            text_ui['background_color'] = style['backgroundColor']
            text_ui['border_style'] = [style['borderTopWidth'], style['borderTopStyle']]
            text_ui['first_option'] = page.locator('#modelPanel .opt').first.text_content()
            assert text_ui['dropdown_open'] and text_ui['item_count'] > 1
            assert text_ui['item_count'] == text_count, (text_ui['item_count'], text_count)
            assert text_ui['trigger_panel_right_delta'] <= 2, text_ui['trigger_panel_right_delta']
            assert text_ui['opaque_background'], 'panel must be opaque'
            assert text_ui['visible_border'], 'panel must have a visible border'
            page.screenshot(path=EVIDENCE/'text-model-dropdown.png')
            artifacts['text-model-dropdown'] = dict(
                kind='text-model-dropdown', path='text-model-dropdown.png',
                sha256=_sha256(EVIDENCE/'text-model-dropdown.png'), ui=text_ui)

            # ---- multimodal-model-dropdown.png: image-capable models only ---
            page.locator('#modality').select_option('image')
            page.wait_for_function(
                "() => /live catalog|no model|Catalog unavailable/.test("
                "document.getElementById('catalogMeta').textContent)", timeout=60000)
            image_count = int(page.evaluate('window.__orca.state().models'))
            image_meta = page.locator('#catalogMeta').text_content()
            multi_ui = ui['multimodal-model-dropdown'] = {}
            if image_count:
                page.locator('#modelTrigger').click()
                page.wait_for_selector('#modelPanel .opt')
                multi_ui['dropdown_open'] = panel.is_visible()
                multi_ui['item_count'] = page.locator('#modelPanel .opt').count()
                multi_ui['opaque_background'] = ('rgba(0, 0, 0, 0)'
                                                 != style['backgroundColor'])
                multi_ui['visible_border'] = text_ui['visible_border']
                multi_ui['trigger_panel_right_delta'] = text_ui['trigger_panel_right_delta']
                assert multi_ui['item_count'] == image_count, (multi_ui['item_count'],
                                                               image_count)
                # Every option that reaches the selector must declare image input.
                declared = page.evaluate("""async () => {
                    const r = await fetch('/api/models?capability=chat&modality=image',
                        {cache:'no-store'});
                    const d = await r.json();
                    return d.models.map(m => [m.id, (m.input_modalities||[]).join(',')]);
                }""")
                multi_ui['models'] = [row[0] for row in declared]
                for identifier, modalities in declared:
                    assert 'image' in modalities, f'{identifier} lacks declared image input'
                text_only = page.evaluate("""async () => {
                    const r = await fetch('/api/models?capability=chat', {cache:'no-store'});
                    const d = await r.json();
                    return d.models.map(m => m.id);
                }""")
                text_ui['models'] = text_only
                assert set(multi_ui['models']).issubset(set(text_only))
                assert set(multi_ui['models']) != set(text_only), \
                    'the image filter must actually narrow the list'
                page.screenshot(path=EVIDENCE/'multimodal-model-dropdown.png')
            else:
                multi_ui['item_count'] = 0
                multi_ui['dropdown_open'] = False
                multi_ui['note'] = ('workspace catalog declares no image-input chat model; '
                                    'the multimodal selector correctly shows none')
                page.screenshot(path=EVIDENCE/'multimodal-model-dropdown.png')
            artifacts['multimodal-model-dropdown'] = dict(
                kind='multimodal-model-dropdown', path='multimodal-model-dropdown.png',
                sha256=_sha256(EVIDENCE/'multimodal-model-dropdown.png'), ui=multi_ui)

            assert not errors, errors
        finally:
            browser.close()

    # Screenshots must meet the minimum evidence size.
    from PIL import Image
    for record in artifacts.values():
        with Image.open(EVIDENCE/record['path']) as image:
            assert image.width >= MIN_WIDTH and image.height >= MIN_HEIGHT, record['path']
            record['size'] = [image.width, image.height]
    ui['catalog_meta'] = image_meta

    manifest = dict(
        automation=dict(framework='playwright', passed=True, catalog_source=CATALOG_SOURCE,
                        catalog_model_count=text_count, image_model_count=image_count,
                        runner='python3 -m pytest hybrid_rollout/robodojo/'
                               'test_orcarouter_console_browser.py'),
        artifacts=[artifacts[kind] for kind in
                   ('auth-methods', 'text-model-dropdown', 'multimodal-model-dropdown')],
        ui=ui,
        notes=('Real console process, real live OrcaRouter catalog, real masked secret. '
               'The credential is stored server side and is never sent to the browser.'),
    )
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(dict(event='orca_gui_evidence', text_models=text_count,
                          image_models=image_count, page_errors=errors), sort_keys=True))
