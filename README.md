# GPT 6 Astra as an Embodied Policy

Evaluation code and a bilingual technical report by Jiayi Su, Yixin Zheng, Mi Yan, Li Yi, Zhizheng Zhang, and He Wang. Jiayi Su and Yixin Zheng contributed equally; Li Yi, Zhizheng Zhang, and He Wang are corresponding authors.

- [Report and complete video gallery](https://anonymous-report-421.github.io/public-website/)
- [Production website repository](https://github.com/anonymous-report-421/public-website)
- [Citation](CITATION.bib)

We evaluate GPT 6 Astra Direct and a hybrid policy that reviews and optionally corrects π₀.₅ actions on ten RoboDojo tasks, with five aligned cases per task. The hybrid policy achieves a mean Score of **62.60** and **48%** success, with GPT corrections on **14.4% of executed control steps**. GPT 6 Astra Direct achieves **26%** success and a mean Score of **37.81** over the 48 episodes with available native scores. Both success rates use all 50 selected cases. Official-model comparisons are reweighted public references, not same-seed reruns.

## Repository layout

```text
hybrid_rollout/
  robodojo/               simulator, policy server, controller, skills and scheduler
  robolab/                separate RoboLab integration
  report_site/            bilingual report source, gallery and packaging tools
  assets/                 licensed visualization fonts
report_web/               prebuilt report and its 21 article clips
public_results/           selected case metadata, scores, seeds and provenance
licenses/                 third-party notices
```

This is a history-free source release. It contains no credentials, private annotations, raw model sessions, original operations logs, simulator assets or model checkpoints. The website repository hosts 100 RoboDojo evaluation videos and 43 gallery clips. The report contains one extra historical demonstration clip used only for qualitative analysis. The [RoboLab gallery](https://anonymous-report-421.github.io/public-website/robolab-gallery.html) contains 150 recordings (three methods, ten tasks, five trials); its unchanged original MP4s are hosted in the website repository's [video release](https://github.com/anonymous-report-421/public-website/releases/tag/robolab-gallery-20260914) to keep the static Pages site below its size limit. The gallery manifest records each video's URL, size and SHA-256.

## Preview the report

From the repository root:

```sh
python3 -m hybrid_rollout.report_site.preview --directory report_web --port 8768
```

Open `http://127.0.0.1:8768/`. The preview server supports video byte ranges. English is the default; the language button or `?lang=zh` switches to Chinese. The static `report_web/` folder can also be hosted on a standard HTTP server.

## Rebuild the report

Requires Node.js compatible with the locked Vite version (Node 24 was used for this release).

```sh
cd hybrid_rollout/report_site/app
npm ci --ignore-scripts --no-audit --no-fund
npm run build
```

The build verifies the preserved shared runtime and produces a self-contained `dist/index.html`. Copy that file to `report_web/index.html`, keeping `data.json`, `scores.csv`, `citation.bib`, licenses and media alongside it. The public authoring snapshot is included; no private experiment storage is needed for this rebuild. Full gallery media are in the separate website repository. Do not run `build.py` merely to rebuild the frontend: that offline evidence-extraction utility requires the original, non-public trajectory archives.

## Run a new evaluation

The integrations depend on externally installed simulator and policy environments. They are **not** a standalone simulator distribution. RoboDojo uses Isaac Sim 5.1, the matching RoboDojo source/assets and its π₀.₅ OpenPI/JAX checkpoint. RoboLab has its own environment. Obtain these from their upstream projects under their own terms. See `hybrid_rollout/robodojo/SOURCE.json` for the recorded upstream revisions and checkpoint identity.

1. Prepare your simulator environment, OpenPI/JAX environment, checkpoint and Codex CLI. Keep these separate from the lightweight tools environment.
2. Review `hybrid_rollout/robodojo/cluster.env.example` and set your paths, image, user/group IDs, credentials file paths and explicit case identity. Public export paths are generic examples, not runnable cluster credentials.
3. Use `hybrid_rollout.robodojo.codex_backend.profiles` to manage **isolated** rollout authentication. Never commit keys or `auth.json`, and do not modify your interactive Codex home. Inspect the CLI with `--help` before performing login or lease operations.
4. The container entry is `hybrid_rollout/robodojo/cluster_entrypoint.sh`; `acp.py` is an optional SenseCore-specific submission adapter. A container runs one explicit case, saves artifacts, and cleans up its own components. Cluster-specific campaign migration helpers require deployment-local plans and are not a one-command public launch recipe.
5. Use `public_results/evaluation_cases.json` to align task, scene and seed across policies. Each case has one hybrid and one Direct result. Set `ROLLOUT_EVALUATION_METHOD=pi05_plus_gpt` or `gpt_only`; the latter is an internal CLI identifier for GPT 6 Astra Direct and does not launch π₀.₅.

The model is fixed to `gpt-6-astra` with `xhigh` reasoning. No provider fallback is implied. You need your own authorized account or gateway access. Policy/model execution can incur costs. Read the skills and action contract before launching; no simulation or model calls are made by the preview or offline tests.

### OrcaRouter provider

OrcaRouter is available as a first-class provider with two explicit
authentication choices that share one inference identity:
`ROLLOUT_AUTH_PROFILE=orcarouter` (`OrcaRouter - API`, paste an `sk-orca-…` key)
and `ROLLOUT_AUTH_PROFILE=orcarouter_oauth` (`OrcaRouter - Auth`, sign in with
OAuth 2.0 + PKCE). Both route inference to `https://api.orcarouter.ai/v1` with
`wire_api = "responses"`; authentication uses the separate
`https://www.orcarouter.ai` origin. Credentials are stored in the existing
private rollout store under `private/`, and a local console shows both choices
side by side with a model picker built from the live catalog. See
[`hybrid_rollout/robodojo/codex_backend/orcarouter/README.md`](hybrid_rollout/robodojo/codex_backend/orcarouter/README.md).

### Deployment and security boundaries

Gateway URLs in `codex_backend/profiles.py` and `codex_backend/config.toml` use reserved `.invalid` placeholders. Configure your own authorized endpoint explicitly in the selected profile and matching configuration, and set `ROLLOUT_GATEWAY_NO_PROXY` for your network if needed. Historical profile identifiers are retained only for compatibility; no company or relay service is configured by this release. The optional SenseCore adapter uses that platform's public control-plane endpoints, not credentials.

The annotation/review tools are intended for a trusted local network, not unauthenticated Internet hosting. Keep the default loopback binding, or put an authenticated reverse proxy in front of any LAN-accessible service. Public GitHub Pages serves only the compiled static report and selected media, never the annotation API or rollout services. Codex tool execution and simulator RPC belong in an isolated, trusted environment; do not supply untrusted programs, serialized RPC data or trajectory archives. Store credentials outside this repository, with restricted permissions. The checked-in token/proxy examples in tests are synthetic fixtures.

## Offline tests

Install `requirements-tools.txt` in a dedicated virtual environment. Simulator and OpenPI dependencies must be installed separately in their respective environments.

```sh
python3 -m unittest hybrid_rollout.report_site.test_i18n hybrid_rollout.report_site.test_editorial_revision -v
python3 -m pytest hybrid_rollout/robodojo/test_proposal_diagnostics.py hybrid_rollout/robodojo/test_prompt_context.py hybrid_rollout/robodojo/test_network_continue.py
```

Tests that require original shared storage, a proprietary simulator or historical operations fixtures cannot be reproduced solely from this public snapshot. The release has not rerun paid model or GPU experiments.

## Licensing and citation

Project-owned code is MIT licensed. Third-party components, fonts, benchmark assets and logos retain their own terms; see `THIRD_PARTY_NOTICES.md`. Model weights and simulator assets are not redistributed.

If this report or code is useful to your research, please cite:

```bibtex
@misc{su2026astra,
  title        = {{GPT 6 Astra} as an Embodied Policy},
  author       = {Su, Jiayi and Zheng, Yixin and Yan, Mi and Yi, Li and Zhang, Zhizheng and Wang, He},
  year         = {2026},
  howpublished = {Technical report and code},
  url          = {https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy}
}
```
