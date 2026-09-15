# Licensing scope and third-party notices

The root MIT license applies to code and documentation owned by Yu-Mool Shu and Lipxin Zheng. It does not relicense third-party components, benchmark assets, model weights, fonts, logos or cited material.

- **Data report runtime:** the copied shared report application is identified by `hybrid_rollout/report_site/app/protected-runtime.json`. Its existing attribution and integrity files are retained. Third-party dependency notices are under `licenses/`; project-owned report content is under `app/src/content/report/`. The root MIT grant is not a new license grant for upstream-owned runtime files.
- **JavaScript dependencies:** the exact dependency resolution is in the report app's `package-lock.json`. Bundled dependency licenses are retained under `licenses/frontend/` and the runtime notice file. Dependencies are not vendored as `node_modules` in this release.
- **Fonts:** Noto CJK/Serif and Source Serif font notices are retained with the font assets and under the prebuilt report's `licenses/` directory. Follow the relevant font license when redistributing or modifying them.
- **RoboDojo, OpenPI and RoboLab:** external simulator, benchmark, policy source and checkpoint references are identified in each integration's `SOURCE.json` and code. Obtain these separately under their upstream licenses; their assets and weights are not included.
- **Benchmark/model logos and linked research:** included for attribution and identification, without claiming trademark ownership, endorsement or relicensing.
- **Rollout videos:** recorded outputs of the described evaluation. Underlying benchmark assets and third-party content remain subject to their original rights. The code license does not supersede those rights.

Upstream references: https://robodojo-benchmark.com/ and https://github.com/Physical-Intelligence/openpi.
