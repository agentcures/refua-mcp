# Refua MCP Server

MCP server exposing strict, typed Refua tools for Boltz2 folding/affinity and BoltzGen design workflows.

## Install

```bash
pip install refua[cuda] # remove [cuda] if you don't need gpu support
pip install refua-mcp
```

ADMET predictions are optional; install `refua[admet]` to enable them:

```bash
pip install refua[admet]
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

## Tools

- `refua_validate_spec`: validate and normalize a request without running folding/affinity.
- `refua_fold`: run fold/design workflows with typed entities and constraints.
- `refua_affinity`: run affinity-only predictions.
- `refua_antibody_design`: focused antibody entrypoint (`antibody` + optional `context_entities`).
- `refua_job`: check status/results for background jobs.
- `refua_admet_profile` (optional): run model-based ADMET predictions for SMILES strings (requires `refua[admet]`).

All major tools expose strict JSON schemas, including discriminated entity unions by `type` and typed output schemas.

## Resources And Templates

- `refua://recipes/index` (resource): lists canonical recipe names.
- `refua://recipes/{recipe_name}` (resource template): returns concrete tool/args examples.

Recipe names:
- `fold_protein_ligand`
- `affinity_only`
- `antibody_design`

## Workflow

Recommended call sequence:

1. Read `refua://recipes/index` and optionally a recipe template.
2. Call `refua_validate_spec` to catch schema/logic issues before expensive runs (`deep_validate=true` for asset-backed construction checks).
3. Execute `refua_fold`, `refua_affinity`, or `refua_antibody_design`.
4. For long runs, set `async_mode=true` and poll `refua_job`.

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

Long-poll support:

```json
{
  "tool": "refua_job",
  "args": {
    "job_id": "...",
    "wait_for_terminal_seconds": 300,
    "include_result": true
  }
}
```

## Experimental MCP Tasks

This server enables MCP experimental task support (`tasks/get`, `tasks/result`,
`tasks/list`, `tasks/cancel`) and advertises task execution support for
`refua_fold`, `refua_affinity`, `refua_antibody_design`, and `refua_admet_profile`.

If your client supports task-augmented tool calls, prefer tasks for long-running
operations. Otherwise, continue with `async_mode=true` + `refua_job`.
