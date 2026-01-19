# Refua MCP Server

MCP server exposing Refua Boltz2 folding/affinity and BoltzGen antibody/peptide design helpers.

## Install

```bash
pip install refua[cuda] # remove [cuda] if you don't need gpu support
pip install refua-mcp
```

Boltz2 and BoltzGen require model/molecule assets. If you don't have them, refua can download them for you automatically:

```bash
python -c "from refua import download_assets; download_assets()"
```

- Boltz2: uses `~/.boltz` by default. Override via tool `cache_dir` if needed.
- BoltzGen: uses the bundled HF artifact by default. Override via tool `mol_dir` if needed.

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

## Tools

- `boltz2_fold_complex`: fold a complex and return a structure in mmCIF/BCIF (set `async_mode=true` to enqueue).
- `boltz2_affinity`: predict affinity for a ligand binder (set `async_mode=true` to enqueue).
- `boltz2_job`: check status for background Boltz2 jobs and optionally return results.
- `boltzgen_antibody_design`: build BoltzGen antibody design features.
- `boltzgen_peptide_design`: build BoltzGen peptide binder design features.

## Long-Running Jobs

For runs that exceed the tool-call timeout, set `async_mode=true` and poll the job:

```json
{
  "tool": "boltz2_affinity",
  "args": {
    "async_mode": true,
    "chains": [...]
  }
}
```

Then poll with:

```json
{
  "tool": "boltz2_job",
  "args": {
    "job_id": "..."
  }
}
```

Set `include_result=true` once the job is complete to fetch the output.
