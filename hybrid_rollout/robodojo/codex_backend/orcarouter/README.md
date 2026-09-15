# OrcaRouter provider

OrcaRouter is a first-class rollout provider in this repository. It appears in
the profile registry as **two explicit authentication choices** that share one
inference identity, so nothing downstream needs to know how the user signed in.

| Profile id | Label | Credential source | Stored at (under the shared root) |
| --- | --- | --- | --- |
| `orcarouter` | `OrcaRouter - API` | user pastes an `sk-orca-…` key | `private/orcarouter.key` |
| `orcarouter_oauth` | `OrcaRouter - Auth` | OAuth 2.0 + PKCE sign-in issues the same kind of key | `private/orcarouter_oauth.key` |

Both entries emit the identical `[model_providers.*]` block:

```toml
[model_providers.orcarouter]
name = "OrcaRouter"
base_url = "https://api.orcarouter.ai/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
```

```sh
ROLLOUT_AUTH_PROFILE=orcarouter python3 -m hybrid_rollout.robodojo.codex_backend.validate <config.toml>
```

## Why two ids instead of one button

The two choices fail differently and are logged out differently. A single
"OrcaRouter" control that sometimes prompts for a key and sometimes opens a
browser cannot report *which* credential is broken when a request is rejected.
Keeping the ids separate also lets a support answer be specific: "your pasted
key was revoked" versus "your account sign-in needs renewing".

## Connect flows

| Flow | Used by | Why |
| --- | --- | --- |
| **A — loopback redirect** | the local provider console | it is itself a loopback HTTP service, so it can receive the redirect: one click, nothing to paste |
| **B — out-of-band code** | the CLI (`orcarouter-connect --flow oob`) | a rollout install address differs on every deployment, and there is no callback URL to pre-register |
| **C — device grant** | `--flow device` (optional extra) | headless hosts; never a substitute for PKCE |

Every attempt sends `code_challenge_method=S256`. Flow A included: the consent
screen lets a user ask for a displayed code, and a displayed code must be
redeemable only by the process holding the verifier. No client secret is
involved and no redirect URI has to be registered.

## Command line

```sh
# OrcaRouter - API: store a key (from a private file, or ORCA_KEY in the environment)
python3 -m hybrid_rollout.robodojo.codex_backend.profiles orcarouter-set-key --key-file ~/.orca.key

# OrcaRouter - Auth: sign in
python3 -m hybrid_rollout.robodojo.codex_backend.profiles orcarouter-connect --flow oob

# Status of both choices (only a masked form is ever printed)
python3 -m hybrid_rollout.robodojo.codex_backend.profiles orcarouter-status

# The live catalog, filtered per entry point
python3 -m hybrid_rollout.robodojo.codex_backend.profiles orcarouter-models --capability chat
python3 -m hybrid_rollout.robodojo.codex_backend.profiles orcarouter-models --capability chat --modality image

# Local console (both choices side by side)
python3 -m hybrid_rollout.robodojo.codex_backend.profiles orcarouter-console
```

## Local provider console

`orcarouter-console` serves a loopback-only page in the same idiom as
`hybrid_rollout.dashboard`: standard library only, `Host`-header rebinding
guard, CSRF token on every write. It shows the two authentication choices side
by side, each with its own status and masked credential, plus a model picker
populated from the real catalog.

The API key never reaches the browser. Model discovery runs server side with the
stored credential and only minimal, non-secret metadata (id, name, owner,
context length, input modalities) is sent to the page.

## Origins and configuration

Authentication and inference live on **different public origins**:

- auth and code exchange — `https://www.orcarouter.ai` (`/auth`,
  `/api/v1/auth/keys`, `/api/v1/auth/device/*`)
- inference and model discovery — `https://api.orcarouter.ai/v1`

Neither is derived from the other. `https://api.orcarouter.ai/v1/auth/keys` is a
404 on the relay and is never produced.

| Variable | Effect |
| --- | --- |
| `ORCA_AUTH_BASE_URL` | explicit authentication origin (highest precedence) |
| `ORCA_API_BASE_URL` | explicit inference origin (highest precedence) |
| `ORCA_BASE_URL` | shared self-hosted base, used for both when neither override is set |

An explicit override always wins. Remote origins must be HTTPS; HTTP is accepted
only for loopback. A base carrying a path (such as `/v1`) is refused rather than
silently producing a doubled prefix.

## Credential lifecycle

A PKCE sign-in returns a **durable** OrcaRouter API key, not an access/refresh
pair. There is no refresh grant, and re-authorizing on every launch is wrong:
the consent endpoint allows ten issued keys per user per 24 hours. The stored
key is reused until it is revoked.

A relay `401` is terminal for the credential that made the request:

- the exact account and the exact credential **generation** are marked
  `needs_reauth`; a late failure from a superseded credential cannot mark a
  freshly re-authorized one;
- no refresh is attempted;
- the old secret is **not** deleted before a replacement login succeeds, so a
  transient failure never becomes an account loss.

Users can revoke every key issued to this app from
<https://www.orcarouter.ai/console/authorized-apps>.

## Model catalog

The single source of truth is `GET <api_base>/v1/models`, optionally narrowed by
`?capability=`. Model ids keep their `vendor/model` namespace verbatim.

Each entry point filters independently and **fails closed**: a model that does
not declare a required input modality is excluded rather than admitted by
default.

| Entry point | Filter |
| --- | --- |
| text chat / agent | `?capability=chat`, an OpenAI/Anthropic/Gemini endpoint type, and no media-only endpoint |
| multimodal understanding | the chat rules **plus** `architecture.input_modalities` containing the modality actually attached |
| embedding | `?capability=embedding` |
| image generation | `?capability=image` |
| video generation | `?capability=video` |
| rerank | `?capability=rerank` |

Live discovery is authoritative. When it fails, a small verified seed
(`openai/gpt-5.5`, `anthropic/claude-opus-4.8`, `google/gemini-3.5-flash`,
`deepseek/deepseek-v4-pro`, `orcarouter/auto`) keeps a fresh installation usable
and is reported as degraded. The seed is never merged into a successful live
result. Verified metadata — context length, input modalities, and the
`low`/`medium`/`high`/`xhigh` reasoning ladder for `openai/gpt-5.5` — is
preserved rather than flattened by discovery.

Selecting a different entry point, or adding an attachment the selected model
does not declare, re-filters the options and clears a selection that is no
longer compatible instead of silently keeping it.

## Tests

```sh
python3 -m pytest hybrid_rollout/robodojo/test_orcarouter_provider.py
python3 -m pytest hybrid_rollout/robodojo/test_orcarouter_console_browser.py
```

The provider suite runs entirely against local fake origins and never contacts
OrcaRouter. Tests whose names contain `live` exercise the real service and are
skipped unless `ORCAROUTER_API_KEY` is set. The browser suite additionally
requires a Playwright runtime and a Chromium binary, and writes real screenshots
plus `orca-evidence/manifest.json`. That directory is generated, ignored by git,
and attached to the pull request as an evidence bundle rather than committed, so
the change stays source and tests only.
