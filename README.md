# Refua MCP Server

MCP server exposing strict, typed Refua tools for Boltz2 folding/affinity and BoltzGen design workflows.

Protocol target: MCP spec revision `2025-11-25`.

## Install

```bash
pip install refua[cuda] # remove [cuda] if you don't need gpu support
pip install refua-mcp
```

ADMET predictions are optional; install `refua[admet]` to enable them:

```bash
pip install refua[admet]
```

Clinical trial simulation is optional; install the clinical extra:

```bash
pip install "refua-mcp[clinical]"
```

Preclinical planning/scheduling/bioanalysis/CMC tools are optional; install the preclinical extra:

```bash
pip install "refua-mcp[preclinical]"
```

OpenTelemetry spans/metrics are optional:

```bash
pip install "refua-mcp[observability]"
```

Boltz2 and BoltzGen require model/molecule assets. If you don't have them, refua can download them for you automatically:

```bash
python -c "from refua import download_assets; download_assets()"
```

- Boltz2: uses `~/.boltz` by default. Override via tool `boltz.cache_dir` if needed.
- BoltzGen: uses the bundled HF artifact by default. Override via tool `boltzgen.mol_dir` if needed.

## MCP Clients

### Claude Code

Add the server to your Claude Code MCP config (macOS: `~/Library/Application Support/Claude/claude_code_config.json`, Linux: `~/.config/claude/claude_code_config.json`). This uses the default assets (`~/.boltz` for Boltz2 and the bundled BoltzGen artifact). Merge with any existing `mcpServers` entries:

```json
{
  "mcpServers": {
    "refua-mcp": {
      "command": "python3",
      "args": ["-m", "refua_mcp.server"]
    }
  }
}
```

### Codex

Register the server with the Codex CLI (uses default asset locations):

```bash
codex mcp add refua-mcp -- python3 -m refua_mcp.server
```

List configured servers with:

```bash
codex mcp list
```

If the server is slow to boot (for example on first import of heavy ML deps),
raise the startup timeout in your Codex `config.toml`:

```toml
[mcp_servers.refua-mcp]
startup_timeout_sec = 30
```

## Transport And Security

Default transport is `stdio`. You can run HTTP transports with runtime env vars:

```bash
REFUA_MCP_TRANSPORT=streamable-http python -m refua_mcp.server
```

Supported transport modes:
- `REFUA_MCP_TRANSPORT=stdio|sse|streamable-http`
- `REFUA_MCP_HOST` / `REFUA_MCP_PORT`
- `REFUA_MCP_MOUNT_PATH`

Security/runtime controls:
- `REFUA_MCP_ENABLE_DNS_REBINDING_PROTECTION=true|false`
- `REFUA_MCP_ALLOWED_HOSTS=host1,host2`
- `REFUA_MCP_ALLOWED_ORIGINS=https://origin1,https://origin2`
- `REFUA_MCP_AUTH_TOKENS=token1,token2` (static bearer tokens for HTTP transports)
- `REFUA_MCP_TASK_TIMEOUT_SECONDS`
- `REFUA_MCP_QUEUE_TIMEOUT_SECONDS`

## Tools

- `refua_validate_spec`: validate and normalize a request without running folding/affinity.
- `refua_fold`: run fold/design workflows with typed entities and constraints.
- `refua_affinity`: run affinity-only predictions.
- `refua_antibody_design`: focused antibody entrypoint (`antibody` + optional `context_entities`).
- `refua_protein_properties`: compute sequence-based protein properties via Refua's `ProteinProperties` API.
- `refua_job`: check status/results for background jobs.
- `refua_admet_profile` (optional): run model-based ADMET predictions for SMILES strings (requires `refua[admet]`).
- `refua_clinical_simulator` (optional): run the `refua-clinical` simulator and optional workup (requires `refua-mcp[clinical]`).
- `refua_preclinical_templates` (optional): fetch default study + sample templates and curated references (requires `refua-mcp[preclinical]`).
- `refua_preclinical_plan` (optional): generate GLP tox/pharmacology study planning payloads.
- `refua_preclinical_schedule` (optional): generate in vivo operational schedules.
- `refua_preclinical_bioanalysis` (optional): run bioanalytical ETL/QC and simple NCA summaries.
- `refua_preclinical_workup` (optional): generate integrated preclinical workups with optional bioanalysis.
- `refua_preclinical_cmc_templates` (optional): fetch CMC starter templates and references.
- `refua_preclinical_cmc_plan` (optional): generate formulation/process/QbD/validation CMC planning outputs.
- `refua_preclinical_batch_record` (optional): generate electronic batch record templates.
- `refua_preclinical_stability_plan` (optional): generate stability study schedules.
- `refua_preclinical_stability_assess` (optional): assess stability results against criteria.
- `refua_preclinical_release_assess` (optional): evaluate release decisions from batch/stability evidence.

All major tools expose strict JSON schemas, including discriminated entity unions by `type` and typed output schemas.

Output-format notes:
- `structure_output_format`: canonical values are `cif` or `bcif`; `mmcif` is accepted as a compatibility alias for `cif`.
- `feature_output_format`: use `torch` or `npz` when writing feature files. If `json` is provided with a file output request, the server normalizes it to `npz` and reports a warning.
- If `feature_output_path` ends with `.json`, the server normalizes it to `.npz` for feature file export.
- Fold/design responses include a `warnings` list when compatibility normalization is applied.

Input-compatibility notes:
- For compatibility with some MCP clients, `entities`, `constraints`, `antibody`, and `context_entities` also accept JSON-encoded strings and will be normalized server-side.

Execution-policy notes:
- `refua_fold` and `refua_antibody_design` block exploratory names by default (for example `sanity_*`, `*_probe`, `schema_*`, `smoke_*`) to avoid costly preflight runs.
- If an exploratory run is intentionally needed, set `allow_exploratory_run=true`.
- Asynchronous fold/affinity/design tools accept `queue_timeout_seconds` (optional).

Error-contract notes:
- Tool errors return a structured payload with `error.code`, `error.message`, optional `error.hint`, and `error.retryable`.

## Resources And Templates

- `refua://capabilities` (resource): runtime feature flags, protocol revision, transport/security config summary.
- `refua://recipes/index` (resource): lists canonical recipe names.
- `refua://recipes/{recipe_name}` (resource template): returns concrete tool/args examples.
- `refua://protein-properties/index` (resource): lists available protein property names/groups.
- `refua://protein-properties/group/{group_name}` (resource template): lists property names in a group.
- `refua://protein-properties/property/{property_name}` (resource template): property metadata.

Completion support:
- `refua://recipes/{recipe_name}` completions for recipe names.
- `refua://protein-properties/group/{group_name}` completions for group names.
- `refua://protein-properties/property/{property_name}` completions for property names.

Recipe names:
- `fold_protein_ligand`
- `affinity_only`
- `antibody_design`
- `protein_properties`
- `clinical_simulation` (optional; available when `refua-clinical` is installed)
- `preclinical_plan` (optional; available when `refua-preclinical` is installed)
- `preclinical_workup` (optional; available when `refua-preclinical` is installed)
- `preclinical_cmc_plan` (optional; available when `refua-preclinical` is installed)
- `preclinical_cmc_release` (optional; available when `refua-preclinical` is installed)

## Workflow

Recommended call sequence:

1. Read `refua://recipes/index` and optionally a recipe template.
2. Read `refua://capabilities` for runtime support and limits.
3. For sequence-only property analysis, call `refua_protein_properties`.
4. Call `refua_validate_spec` to catch schema/logic issues before expensive runs (`deep_validate=true` for asset-backed construction checks).
5. Execute `refua_fold`, `refua_affinity`, `refua_antibody_design`, `refua_clinical_simulator` (optional), or the `refua_preclinical_*` tools (optional).
6. For long runs, set `async_mode=true` and poll `refua_job`.

## Examples

Fold protein + ligand with optional affinity:

```json
{
  "tool": "refua_fold",
  "args": {
    "name": "protein_ligand",
    "entities": [
      {"type": "protein", "id": "A", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
      {"type": "ligand", "id": "lig", "smiles": "CCO"}
    ],
    "constraints": [
      {"type": "pocket", "binder": "lig", "contacts": [["A", 5], ["A", 8]]}
    ],
    "affinity": {"binder": "lig"},
    "admet": true
  }
}
```

Affinity-only:

```json
{
  "tool": "refua_affinity",
  "args": {
    "name": "protein_ligand_affinity",
    "entities": [
      {"type": "protein", "id": "A", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
      {"type": "ligand", "id": "lig", "smiles": "CCO"}
    ],
    "binder": "lig"
  }
}
```

Antibody-focused design:

```json
{
  "tool": "refua_antibody_design",
  "args": {
    "name": "ab_design",
    "antibody": {
      "type": "antibody",
      "ids": ["H", "L"],
      "heavy_cdr_lengths": [12, 10, 14],
      "light_cdr_lengths": [10, 9, 9]
    },
    "context_entities": [
      {"type": "protein", "id": "A", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"}
    ]
  }
}
```

Validate only (no run):

```json
{
  "tool": "refua_validate_spec",
  "args": {
    "action": "fold",
    "entities": [
      {"type": "protein", "id": "A", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
      {"type": "ligand", "id": "lig", "smiles": "CCO"}
    ]
  }
}
```

ADMET predictions:

```json
{
  "tool": "refua_admet_profile",
  "args": {
    "smiles": "CCO",
    "include_scoring": true
  }
}
```

Protein properties:

```json
{
  "tool": "refua_protein_properties",
  "args": {
    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "groups": ["basic"],
    "include_catalog": true
  }
}
```

Clinical simulation (optional):

```json
{
  "tool": "refua_clinical_simulator",
  "args": {
    "trial_id": "small_molecule_phase2",
    "replicates": 80,
    "include_workup": true,
    "include_replicates": false
  }
}
```

Preclinical workup (optional):

```json
{
  "tool": "refua_preclinical_workup",
  "args": {
    "seed": 7,
    "lloq_ng_ml": 1.0,
    "use_template_rows_when_missing": true,
    "include_references": true
  }
}
```

Preclinical CMC release assessment (optional):

```json
{
  "tool": "refua_preclinical_release_assess",
  "args": {
    "batch_results": {
      "assay_percent": 99.0,
      "content_uniformity_av": 9.8,
      "dissolution_q30_percent": 92.0,
      "total_impurities_percent": 0.7,
      "water_content_percent": 1.5,
      "appearance_score": 5.0
    },
    "include_references": true
  }
}
```

Note: DNA/RNA entities are supported for Boltz2 folding only (BoltzGen does not accept DNA/RNA entities).

## Long-Running Jobs

For runs that exceed the tool-call timeout, set `async_mode=true` and poll sparingly
(for example every 30-120 seconds) or follow `recommended_poll_seconds` from `refua_job`.

```json
{
  "tool": "refua_fold",
  "args": {
    "async_mode": true,
    "entities": [...]
  }
}
```

Then poll:

```json
{
  "tool": "refua_job",
  "args": {
    "job_id": "..."
  }
}
```

For queued/running jobs, the response includes `recommended_poll_seconds` plus queue
and estimate metadata (`queue_position`, `queue_depth`, `average_runtime_seconds`,
`estimated_start_seconds`, `estimated_remaining_seconds`).
Set `include_result=true` once complete to fetch results.
Set `cancel=true` in `refua_job` to cancel queued/running jobs.

Long-poll support:

```json
{
  "tool": "refua_job",
  "args": {
    "job_id": "...",
    "wait_for_terminal_seconds": 300,
    "cancel": false,
    "include_result": true
  }
}
```

## Experimental MCP Tasks

This server enables MCP experimental task support (`tasks/get`, `tasks/result`,
`tasks/list`, `tasks/cancel`) and advertises task execution support for
`refua_validate_spec`, `refua_fold`, `refua_affinity`, `refua_antibody_design`,
and `refua_admet_profile`. `refua_clinical_simulator` and
`refua_preclinical_templates` / `refua_preclinical_plan` /
`refua_preclinical_schedule` / `refua_preclinical_bioanalysis` /
`refua_preclinical_workup` / `refua_preclinical_cmc_templates` /
`refua_preclinical_cmc_plan` / `refua_preclinical_batch_record` /
`refua_preclinical_stability_plan` / `refua_preclinical_stability_assess` /
`refua_preclinical_release_assess` are also task-enabled when installed.

If your client supports task-augmented tool calls, prefer tasks for long-running
operations. Task cancellation (`tasks/cancel`) is wired to background job cancellation.
Otherwise, continue with `async_mode=true` + `refua_job`.
