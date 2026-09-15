"""Loopback provider console for OrcaRouter; the credential never leaves here.

The console is an ordinary local HTTP service in the same idiom as
``hybrid_rollout.dashboard`` and ``hybrid_rollout.annotation_dashboard``: standard
library only, loopback by default, DNS-rebinding guard on the ``Host`` header,
and a CSRF token on every write.

It exists to make the two authentication choices explicit and to drive model
discovery from the real catalog. Both run **server side**: the API key is stored
in the project's existing private rollout store and is never read back into the
page — the browser only ever receives a masked form and minimal model metadata.
"""
from __future__ import annotations

import argparse
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import catalog, origins, provider
from .credentials import Credential, CredentialError, CredentialStore
from .pkce import PkceCancelled, PkceError, PkceSession

# The official OrcaRouter mark. It is fetched by this server and re-served from
# the console's own loopback origin, so the page issues no third-party request
# from the operator's browser and works behind a proxy. If the fetch fails the
# page falls back to its text label.
LOGO_URL = 'https://www.orcarouter.ai/orca-logo-classic.png'
LOGO_PATH = '/logo.png'
LOGO_MAX_BYTES = 512_000
LOGO_TIMEOUT = 15.0

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OrcaRouter provider</title>
<style>
:root{--bg:#0f172a;--panel:#1e293b;--panel2:#334155;--ink:#e2e8f0;--dim:#94a3b8;--line:#475569;--accent:#38bdf8;--ok:#22c55e;--bad:#f87171}
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink);max-width:960px;margin:24px auto;padding:0 16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:18px}
.brand img{height:28px;width:auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.row{display:flex;gap:16px;flex-wrap:wrap}.row>.card{flex:1 1 340px;margin-bottom:0}
label{display:block;font-size:13px;color:var(--dim);margin-bottom:4px}
input[type=password],input[type=text],select{font:inherit;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink);width:100%}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:var(--panel2);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);color:#04293a;border-color:var(--accent);font-weight:600}
button:disabled{opacity:.5;cursor:not-allowed}
.pill{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.pill.on{color:var(--ok);border-color:var(--ok)}.pill.warn{color:var(--bad);border-color:var(--bad)}
.masked{font-family:ui-monospace,monospace;color:var(--dim)}
.hint{font-size:12px;color:var(--dim);margin-top:6px;min-height:16px}
.err{color:var(--bad)}.ok{color:var(--ok)}
.combo{position:relative;margin-top:6px}
.combo .trigger{width:100%;display:flex;justify-content:space-between;align-items:center;text-align:left}
.combo .panel{position:absolute;left:0;right:0;top:calc(100% + 4px);z-index:20;max-height:280px;overflow:auto;
  background:#0b1220;border:1px solid var(--line);border-radius:8px;box-shadow:0 12px 28px rgba(0,0,0,.55)}
.combo .panel[hidden]{display:none}
.combo .opt{padding:8px 10px;cursor:pointer;border-bottom:1px solid #16233b}
.combo .opt:last-child{border-bottom:none}
.combo .opt:hover,.combo .opt.active{background:#172554}
.combo .opt small{color:var(--dim);display:block}
.combo .empty{padding:10px;color:var(--dim)}
.meta{font-size:12px;color:var(--dim);margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:3px 0;vertical-align:top}
td:first-child{color:var(--dim);width:150px}
</style>
<body>
<div class="brand"><img id="logo" src="''' + LOGO_PATH + '''" alt="OrcaRouter" onerror="this.style.display='none'">
<div><h1>OrcaRouter provider</h1><small>Local console &middot; loopback only &middot; the key stays on this machine</small></div></div>

<div class="row">
  <div class="card" id="apiKeyCard">
    <h2 id="apiKeyHeading">OrcaRouter - API</h2>
    <div>Status: <span id="apiState" class="pill">checking</span></div>
    <div class="masked" id="apiMasked" style="margin:8px 0">no key stored</div>
    <label for="apiKey">Paste an existing <code>sk-orca-…</code> key</label>
    <input id="apiKey" type="password" autocomplete="off" spellcheck="false" placeholder="sk-orca-…">
    <div style="margin-top:8px;display:flex;gap:8px">
      <button id="saveKey" class="primary">Save key</button>
      <button id="clearKey">Clear key</button>
    </div>
    <div class="hint" id="apiHint"></div>
  </div>

  <div class="card" id="oauthCard">
    <h2 id="oauthHeading">OrcaRouter - Auth</h2>
    <div>Status: <span id="oauthState" class="pill">checking</span></div>
    <div class="masked" id="oauthMasked" style="margin:8px 0">no key stored</div>
    <p style="font-size:13px;color:var(--dim);margin:0 0 10px">Sign in with your OrcaRouter account. No client secret, no redirect URI to register; PKCE binds the code to this process.</p>
    <div style="display:flex;gap:8px">
      <button id="connect" class="primary">Connect with OrcaRouter</button>
      <button id="cancel" disabled>Cancel</button>
    </div>
    <div class="hint" id="oauthHint"></div>
    <div class="hint" id="authUrl" style="overflow-wrap:anywhere"></div>
  </div>
</div>

<div class="card" id="modelCard">
  <h2>Model</h2>
  <div class="row" style="align-items:flex-end">
    <div style="flex:0 1 200px"><label for="capability">Entry point</label>
      <select id="capability">
        <option value="chat">Text chat / agent</option>
        <option value="embedding">Embedding</option>
        <option value="image">Image generation</option>
        <option value="video">Video generation</option>
        <option value="rerank">Rerank</option>
      </select></div>
    <div style="flex:0 1 200px"><label for="modality">Attachments</label>
      <select id="modality">
        <option value="">Text only</option>
        <option value="image">Images</option>
        <option value="audio">Audio</option>
        <option value="video">Video</option>
      </select></div>
    <div style="flex:0 0 auto"><button id="refreshModels">Refresh catalog</button></div>
  </div>
  <div class="combo">
    <button class="trigger" id="modelTrigger" aria-haspopup="listbox" aria-expanded="false" disabled>
      <span id="modelLabel">No model selected</span><span aria-hidden="true">&#9662;</span></button>
    <div class="panel" id="modelPanel" role="listbox" hidden></div>
  </div>
  <div class="meta" id="catalogMeta"></div>
</div>

<script>
const el=id=>document.getElementById(id);const csrf=%TOKEN%;
let generation=0,busy=false,session=null,pollTimer=null,models=[],selected=null,panelOpen=false,bootstrapped=false;
async function api(path,body,opts){opts=opts||{};
  const r=await fetch(path,{method:body===undefined?'GET':'POST',cache:'no-store',
    headers:{'X-Review-Token':csrf,'Content-Type':'application/json'},
    body:body===undefined?undefined:JSON.stringify(body),keepalive:!!opts.keepalive});
  const data=await r.json().catch(()=>({error:'Bad response'}));
  if(!r.ok)throw Error(data.error||('HTTP '+r.status));return data;}
function setBusy(on){busy=on;el('connect').disabled=on;el('cancel').disabled=!on;el('saveKey').disabled=on;}
function hint(id,text,cls){const e=el(id);e.textContent=text||'';e.className='hint'+(cls?' '+cls:'');}
function render(){el('modelTrigger').disabled=!models.length;
  el('modelLabel').textContent=selected?selected.id:(models.length?'Select a model…':'No compatible model');
  el('modelPanel').hidden=!panelOpen;}
function renderPanel(){const p=el('modelPanel');p.replaceChildren();
  if(!models.length){const d=document.createElement('div');d.className='empty';
    d.textContent='No model matches this entry point.';p.append(d);return;}
  for(const m of models){const d=document.createElement('div');d.className='opt'+(selected&&selected.id===m.id?' active':'');
    d.setAttribute('role','option');const s=document.createElement('small');
    s.textContent=[m.owned_by||'',m.context_length?m.context_length+' ctx':'',(m.input_modalities||[]).join('+')].filter(Boolean).join(' · ');
    d.append(document.createTextNode(m.id),s);d.onclick=()=>choose(m);p.append(d);}}
function choose(m){selected=m;panelOpen=false;render();renderPanel();
  hint('oauthHint','');hint('apiHint','');}
function setStatus(which,s){el(which+'State').textContent=s.configured?(s.needs_reauth?'needs reauth':'connected'):'not configured';
  el(which+'State').className='pill '+(s.configured?(s.needs_reauth?'warn':'on'):'');
  el(which+'Masked').textContent=s.secret_masked||'no key stored';}
async function status(){const gen=generation;try{const d=await api('/api/status');
  if(gen!==generation)return;setStatus('api',d.api_key);setStatus('oauth',d.oauth);
  if(!bootstrapped){bootstrapped=true;
    if(d.api_key.configured||d.oauth.configured)loadModels();
    else el('catalogMeta').textContent='Connect an API key or sign in to load the model catalog.';}
  }catch(e){if(gen===generation){hint('apiHint',String(e),'err');}}}
async function loadModels(){const gen=++generation;const cap=el('capability').value,mod=el('modality').value;
  el('catalogMeta').textContent='Loading…';panelOpen=false;render();
  try{const d=await api('/api/models?capability='+encodeURIComponent(cap)+(mod?'&modality='+encodeURIComponent(mod):''));
    if(gen!==generation)return;
    const before=selected;models=d.models||[];selected=null;
    let note=d.degraded?('catalog degraded — showing the verified fallback'+(d.reason?' ('+d.reason+')':'')):('live catalog · '+models.length+' compatible model(s)');
    if(before){selected=models.find(m=>m.id===before.id)||null;
      if(before&&!selected)note+=' · previous selection '+before.id+' is no longer compatible and was cleared';}
    el('catalogMeta').textContent=note;renderPanel();render();
  }catch(e){if(gen===generation){models=[];selected=null;renderPanel();render();
    el('catalogMeta').textContent='Catalog unavailable: '+e.message;}}}
el('capability').onchange=loadModels;el('modality').onchange=loadModels;
el('refreshModels').onclick=loadModels;
el('modelTrigger').onclick=()=>{if(!models.length)return;panelOpen=!panelOpen;render();};
document.addEventListener('click',e=>{if(!e.target.closest('.combo')&&panelOpen){panelOpen=false;render();}});
el('saveKey').onclick=async()=>{const gen=generation;hint('apiHint','Saving…');
  try{const d=await api('/api/key',{key:el('apiKey').value});
    if(gen!==generation)return;el('apiKey').value='';hint('apiHint','Saved to the private rollout store.','ok');
    setStatus('api',d.api_key);loadModels();
  }catch(e){if(gen===generation)hint('apiHint',e.message,'err');}};
el('clearKey').onclick=async()=>{const gen=generation;hint('apiHint','');
  try{const d=await api('/api/key/clear',{});if(gen!==generation)return;
    setStatus('api',d.api_key);hint('apiHint','Stored key cleared.','ok');
  }catch(e){if(gen===generation)hint('apiHint',e.message,'err');}};
el('connect').onclick=async()=>{const gen=++generation;setBusy(true);hint('oauthHint','Waiting for your browser…');
  try{const d=await api('/api/connect/start',{});
    if(gen!==generation)return;session=d.session;el('authUrl').textContent=d.authorize_url;
    window.open(d.authorize_url,'_blank');poll(gen);
  }catch(e){if(gen===generation){setBusy(false);hint('oauthHint',e.message,'err');}}};
function poll(gen){clearTimeout(pollTimer);pollTimer=setTimeout(async()=>{
  if(gen!==generation)return;
  try{const d=await api('/api/connect/poll?session='+encodeURIComponent(session||''));
    if(gen!==generation)return;
    if(d.state==='connected'){setBusy(false);session=null;el('authUrl').textContent='';
      hint('oauthHint','Connected; a key was stored.','ok');setStatus('oauth',d.oauth);loadModels();return;}
    if(d.state==='failed'){setBusy(false);session=null;hint('oauthHint',d.error||'Sign-in failed.','err');return;}
    if(d.state==='none'){setBusy(false);session=null;return;}
    poll(gen);
  }catch(e){if(gen===generation){setBusy(false);session=null;hint('oauthHint',String(e),'err');}}},700);}
el('cancel').onclick=async()=>{const gen=++generation;setBusy(false);hint('oauthHint','Login cancelled. Nothing was stored.');
  try{await api('/api/connect/cancel',{session:session},{});}catch(e){}session=null;el('authUrl').textContent='';};
// A back-forward-cache restore does not remount the page, so the generation is
// invalidated and the busy flag and hint are cleared synchronously here rather
// than relying on a guarded finally block that will refuse to run.
addEventListener('pagehide',()=>{generation++;clearTimeout(pollTimer);busy=false;
  el('connect').disabled=false;el('cancel').disabled=true;el('saveKey').disabled=false;
  hint('oauthHint','');el('authUrl').textContent='';
  const payload=JSON.stringify({session:session,token:csrf});
  try{fetch('/api/connect/cancel',{method:'POST',keepalive:true,cache:'no-store',
    headers:{'X-Review-Token':csrf,'Content-Type':'application/json'},body:payload});}
  catch(e){try{navigator.sendBeacon('/api/connect/cancel',new Blob([payload],{type:'application/json'}));}catch(e2){}}
  session=null;});
status();
render();
window.__orca={state:()=>({busy:busy,panelOpen:panelOpen,models:models.length,selected:selected&&selected.id,
  generation:generation,session:session,apiMasked:el('apiMasked').textContent,oauthMasked:el('oauthMasked').textContent})};
</script></body></html>'''


class ConnectManager:
    """Holds at most one in-flight authorization attempt.

    Starting a new one cancels the previous attempt, so switching auth method or
    pressing Connect twice can never leave two listeners racing for one callback.
    """

    def __init__(self, store, auth_base):
        self.store = store
        self.auth_base = auth_base
        self._lock = threading.Lock()
        self._session = None
        self._state = 'none'
        self._error = None
        self._thread = None

    @property
    def active(self):
        return self._session is not None

    def start(self):
        with self._lock:
            self._cancel_locked()
            session = PkceSession(self.auth_base)
            target = session.begin()
            self._session = session
            self._state = 'pending'
            self._error = None
            self._thread = threading.Thread(target=self._run, args=(session,), daemon=True)
            self._thread.start()
            return session.id, target

    def _run(self, session):
        try:
            code = session.wait_for_code()
            from .pkce import exchange_code
            credential = exchange_code(self.auth_base, code, session.attempt.verifier, session.scope)
            session.finish(credential)
            self.store.save(credential)
            with self._lock:
                if self._session is session:
                    self._state = 'connected'
        except PkceCancelled:
            with self._lock:
                if self._session is session:
                    self._state = 'none'
        except PkceError as error:
            with self._lock:
                if self._session is session:
                    self._state = 'failed'
                    self._error = str(error)
        except Exception:
            # Never surface an unexpected exception body: it could carry the
            # authorization response.
            with self._lock:
                if self._session is session:
                    self._state = 'failed'
                    self._error = 'Sign-in failed. Start it again.'
        finally:
            session.release()

    def _cancel_locked(self):
        if self._session is not None:
            self._session.cancel()
            self._session = None
            self._state = 'none'
            self._error = None

    def cancel(self, session_id=None):
        with self._lock:
            self._cancel_locked()
            self._state = 'none'

    def poll(self, session_id=None):
        with self._lock:
            if session_id and self._session is not None and session_id != self._session.id:
                # A stale poll from an abandoned attempt reports nothing.
                return 'none', None
            return self._state, self._error


def _payload(raw):
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode('utf-8'))
        return data if isinstance(data, dict) else {}
    except (ValueError, UnicodeDecodeError):
        return {}


def fetch_logo(timeout=LOGO_TIMEOUT):
    """Fetch the official mark once; returns PNG bytes or ``None``.

    Failure is not an error: the console renders its text label instead, so a
    proxy or an offline machine never blocks the credential UI.
    """
    request = urllib.request.Request(LOGO_URL, headers={'Accept': 'image/png'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(LOGO_MAX_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if len(raw) > LOGO_MAX_BYTES or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        return None
    return raw


def handler(store, manager, *, token=None, env=None, logo=None, discovery_credential=None):
    csrf = token or secrets.token_urlsafe(32)
    page = PAGE.replace('%TOKEN%', json.dumps(csrf)).encode()
    if logo is None:
        logo = fetch_logo()

    class Handler(BaseHTTPRequestHandler):
        server_version = 'orca-console'

        def log_message(self, format, *args):
            # Never log query contents or credential material.
            print(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                  method=self.command, route=urlsplit(self.path).path)), flush=True)

        def headers_out(self, status, kind, size, extra=None):
            self.send_response(status)
            headers = {'Content-Type': kind, 'Content-Length': str(size), 'Cache-Control': 'no-store',
                       'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
                       'Content-Security-Policy': (
                           "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                           "img-src 'self'; object-src 'none'; frame-ancestors 'none'; "
                           "base-uri 'none'; form-action 'none'")}
            for key, value in dict(headers, **(extra or {})).items():
                self.send_header(key, value)
            self.end_headers()

        def respond(self, status, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.headers_out(status, 'application/json; charset=utf-8', len(body))
            if self.command != 'HEAD':
                self.wfile.write(body)

        def trusted_host(self):
            try:
                parsed = urlsplit('http://' + self.headers.get('Host', ''))
                local_ip = self.connection.getsockname()[0]
                parsed.port  # Reject malformed ports too.
                return (not parsed.username and not parsed.password and not parsed.path
                        and not parsed.query and not parsed.fragment
                        and parsed.hostname in ('127.0.0.1', 'localhost', '::1', local_ip))
            except ValueError:
                return False

        def write_token(self, body):
            """Accept the CSRF token from the header or, for keepalive unload
            requests, from the body. The value is page-scoped either way."""
            supplied = self.headers.get('X-Review-Token', '') or str(body.get('token', ''))
            return secrets.compare_digest(supplied, csrf)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self.trusted_host():
                self.respond(403, dict(error='Use the server IP address or a localhost port forward'))
                return
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            arg = lambda key, default='': query.get(key, [default])[0]
            try:
                if parsed.path == '/':
                    self.headers_out(200, 'text/html; charset=utf-8', len(page))
                    if self.command != 'HEAD':
                        self.wfile.write(page)
                    return
                if parsed.path == '/api/status':
                    self.respond(200, dict(
                        api_key=provider.describe(provider.API_PROVIDER, store),
                        oauth=provider.describe(provider.OAUTH_PROVIDER, store),
                        inference_base=origins.api_v1(env),
                        auth_base=origins.auth_base(env)))
                    return
                if parsed.path == '/api/models':
                    self.respond(200, self.models(arg('capability') or 'chat',
                                                  arg('modality'), env))
                    return
                if parsed.path == LOGO_PATH:
                    if logo is None:
                        raise FileNotFoundError('Logo unavailable')
                    self.headers_out(200, 'image/png', len(logo),
                                     extra={'Cache-Control': 'private, max-age=86400'})
                    if self.command != 'HEAD':
                        self.wfile.write(logo)
                    return
                if parsed.path == '/api/connect/poll':
                    state, error = manager.poll(arg('session') or None)
                    self.respond(200, dict(state=state, error=error,
                                           oauth=provider.describe(provider.OAUTH_PROVIDER, store)))
                    return
                raise FileNotFoundError('Not exposed')
            except (OSError, ValueError) as error:
                self.respond(404, dict(error=str(error) or 'Unavailable'))

        def models(self, capability, modality, env=None):
            """Discover models server side so the key never reaches the browser.

            ``discovery_credential`` lets a caller supply the credential used for
            the catalog request independently of the credential being displayed,
            which keeps an evidence run from rendering a real key's masked form.
            """
            required = (modality,) if modality else ()
            credential = discovery_credential or store.load_any()
            if credential is None:
                return dict(models=[], degraded=False, capability=capability,
                            error='No OrcaRouter credential is configured yet.')
            result = catalog.catalog(credential.reveal(), capability, required, env=env)
            # Only minimal, non-secret metadata crosses to the browser.
            return dict(models=[dict(id=m['id'], name=m['name'], owned_by=m['owned_by'],
                                     context_length=m['context_length'],
                                     input_modalities=m['input_modalities'])
                                for m in result['models']],
                        source=result['source'], degraded=result['degraded'],
                        capability=capability, required_modalities=list(required),
                        reason=result.get('reason'), catalog_url=catalog.discovery_url(capability, env))

        def do_POST(self):
            if not self.trusted_host():
                self.respond(403, dict(error='Use the server IP address or a localhost port forward'))
                return
            length = int(self.headers.get('Content-Length') or 0)
            body = _payload(self.rfile.read(length) if length else b'')
            if not self.write_token(body):
                self.respond(403, dict(error='Invalid write token; reload this local console'))
                return
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                self.respond(403, dict(error='Cross-origin writes are not allowed'))
                return
            path = urlsplit(self.path).path
            try:
                if path == '/api/key':
                    self.save_key(body)
                    return
                if path == '/api/key/clear':
                    store.clear('api_key')
                    self.respond(200, dict(api_key=provider.describe(provider.API_PROVIDER, store)))
                    return
                if path == '/api/connect/start':
                    session_id, target = manager.start()
                    self.respond(200, dict(session=session_id, authorize_url=target,
                                           auth_base=origins.auth_base(env)))
                    return
                if path == '/api/connect/cancel':
                    manager.cancel(body.get('session'))
                    self.respond(200, dict(state='none'))
                    return
                raise FileNotFoundError('Not exposed')
            except CredentialError as error:
                # Credential messages are written to be safe to display.
                self.respond(400, dict(error=str(error)))
            except PkceError as error:
                self.respond(400, dict(error=str(error)))
            except (OSError, ValueError) as error:
                self.respond(404, dict(error=str(error) or 'Unavailable'))

        def save_key(self, body):
            key = body.get('key')
            try:
                credential = provider.adapter_for(provider.API_PROVIDER, store).acquire(key)
            except CredentialError as error:
                self.respond(400, dict(error=str(error)))
                return
            described = provider.describe(provider.API_PROVIDER, store)
            # The response never contains the key, only its masked form.
            self.respond(200, dict(api_key=described, saved=credential.generation))

    return Handler


def serve(store=None, host='127.0.0.1', port=0, env=None, discovery_credential=None):
    store = store or CredentialStore()
    manager = ConnectManager(store, origins.auth_base(env))
    server = ThreadingHTTPServer(
        (host, port), handler(store, manager, env=env, discovery_credential=discovery_credential))
    server.daemon_threads = True
    server.orca_manager = manager
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path)
    parser.add_argument('--host', default='127.0.0.1',
                        help='0.0.0.0 exposes an unauthenticated console; trusted networks only')
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    store = CredentialStore(shared=args.shared_root)
    server = serve(store, args.host, args.port)
    print(json.dumps(dict(event='orcarouter_console',
                          url=f'http://{args.host}:{server.server_port}/',
                          inference_base=origins.api_v1(), auth_base=origins.auth_base(),
                          credential_value_exposed=False)), flush=True)
    server.serve_forever(poll_interval=1)


if __name__ == '__main__':
    main()
