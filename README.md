# Refua MCP Server

MCP server exposing Refua's unified Complex API for Boltz2 folding/affinity and BoltzGen design workflows.

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

- `refua_complex`: run a unified Complex spec with `action="fold"` (default) or `action="affinity"`. Supports `protein|dna|rna|binder|peptide|antibody|ligand|file` entities. Optionally run ADMET for SMILES ligands via `admet` (auto by default when available).
- `refua_job`: check status for background jobs and optionally return results.
- `refua_admet_profile` (optional): run model-based ADMET predictions for SMILES strings (only available when `refua[admet]` is installed).

Example (fold a protein + ligand with optional affinity):

```json
{
  "tool": "refua_complex",
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

Example (ADMET predictions):

```json
{
  "tool": "refua_admet_profile",
  "args": {
    "smiles": "CCO",
    "include_scoring": true
  }
}
```

Example (antibody helper + peptide helper):

```json
{
  "tool": "refua_complex",
  "args": {
    "name": "design_helpers",
    "entities": [
      {"type": "protein", "id": "A", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
      {"type": "antibody", "ids": ["H", "L"], "heavy_cdr_lengths": [12, 10, 14], "light_cdr_lengths": [10, 9, 9]},
      {"type": "peptide", "id": "P", "segment_lengths": [10, 6, 3], "disulfide": true}
    ]
  }
}
```

Note: DNA/RNA entities are supported for Boltz2 folding only (BoltzGen does not accept DNA/RNA entities).

## Long-Running Jobs

For runs that exceed the tool-call timeout, set `async_mode=true` and poll the job.
Folding (`action="fold"`) can take minutes depending on inputs and hardware, so poll
sparingly (for example, every 10-30 seconds with backoff) or follow
`recommended_poll_seconds` from `refua_job`.

```json
{
  "tool": "refua_complex",
  "args": {
    "async_mode": true,
    "entities": [...]
  }
}
```

Then poll with:

```json
{
  "tool": "refua_job",
  "args": {
    "job_id": "..."
  }
}
```

For queued/running jobs, the response includes `recommended_poll_seconds` plus queue
and estimate metadata (`queue_position` = jobs ahead, `queue_depth` = queued jobs,
`average_runtime_seconds`, `estimated_start_seconds`, `estimated_remaining_seconds`)
to help clients back off.
Set `include_result=true` once the job is complete to fetch the output.
