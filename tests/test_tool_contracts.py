from __future__ import annotations

import json
from types import SimpleNamespace

import anyio
from mcp.types import CallToolResult

import refua_mcp.server as server


def test_refua_fold_schema_uses_discriminated_entities() -> None:
    tool = next(
        info
        for info in server.mcp._tool_manager.list_tools()
        if info.name == "refua_fold"
    )
    entities_schema = tool.parameters["properties"]["entities"]
    array_variant = entities_schema["anyOf"][0]
    entities_items = array_variant["items"]["anyOf"][0]
    assert "oneOf" in entities_items
    assert entities_items["discriminator"]["propertyName"] == "type"


def test_validate_spec_returns_execution_plan() -> None:
    result = server.refua_validate_spec(
        entities=[
            server.ProteinEntity(type="protein", id="A", sequence="MKTAYIAK"),
            server.LigandEntity(type="ligand", id="lig", smiles="CCO"),
        ],
        action="fold",
    )

    assert result.valid is True
    assert result.execution_plan.action == "fold"
    assert result.execution_plan.run_boltz is True
    assert result.execution_plan.run_boltzgen is False
    assert result.execution_plan.ligand_id_map == {"lig": "L1"}
    assert any("deep_validate=true" in warning for warning in result.warnings)


def test_recipe_template_returns_split_tool_examples() -> None:
    payload = json.loads(server.refua_recipe_template("antibody_design"))
    assert payload["tool"] == "refua_antibody_design"


def test_antibody_design_schema_exposes_output_format_compat_aliases() -> None:
    tool = next(
        info
        for info in server.mcp._tool_manager.list_tools()
        if info.name == "refua_antibody_design"
    )
    structure_enum = set(
        tool.parameters["properties"]["structure_output_format"]["anyOf"][0]["enum"]
    )
    feature_enum = set(
        tool.parameters["properties"]["feature_output_format"]["anyOf"][0]["enum"]
    )
    assert structure_enum == {"cif", "mmcif", "bcif"}
    assert feature_enum == {"torch", "npz", "json"}


def test_resolve_output_format_maps_mmcif_alias_to_cif() -> None:
    assert server._resolve_output_format(None, "mmcif") == "cif"


def test_json_feature_output_format_errors_only_for_file_writes() -> None:
    try:
        server._resolve_feature_output_format("features.pt", "json")
        raise AssertionError("Expected ValueError for json feature file output format.")
    except ValueError as exc:
        assert "inline MCP response summaries" in str(exc)

    result = server.refua_validate_spec(
        entities=[server.ProteinEntity(type="protein", id="A", sequence="MKTAYIAK")],
        feature_output_format="json",
    )
    assert any(
        "ignored unless feature_output_path" in warning for warning in result.warnings
    )


def test_validate_spec_normalizes_json_feature_file_output_request() -> None:
    result = server.refua_validate_spec(
        entities=[server.ProteinEntity(type="protein", id="A", sequence="MKTAYIAK")],
        feature_output_path="design_features.json",
        feature_output_format="json",
    )
    assert result.normalized_input["feature_output_format"] == "npz"
    assert result.normalized_input["feature_output_path"].endswith(
        "design_features.npz"
    )
    assert any("normalized to 'npz'" in warning for warning in result.warnings)


def test_validate_spec_normalizes_json_feature_path_without_format() -> None:
    result = server.refua_validate_spec(
        entities=[server.ProteinEntity(type="protein", id="A", sequence="MKTAYIAK")],
        feature_output_path="design_features.json",
    )
    assert result.normalized_input["feature_output_format"] == "npz"
    assert result.normalized_input["feature_output_path"].endswith(
        "design_features.npz"
    )
    assert any("ending in '.json'" in warning for warning in result.warnings)


def test_validate_spec_normalizes_mmcif_alias() -> None:
    result = server.refua_validate_spec(
        entities=[server.ProteinEntity(type="protein", id="A", sequence="MKTAYIAK")],
        structure_output_format="mmcif",
    )
    assert result.normalized_input["structure_output_format"] == "cif"
    assert any("normalized to 'cif'" in warning for warning in result.warnings)


def test_refua_fold_normalizes_output_requests_before_execution() -> None:
    captured: dict[str, object] = {}
    original = server._run_complex_operation

    def fake_run_complex_operation(**kwargs):
        captured.update(kwargs)
        return {
            "name": kwargs["name"],
            "backend": "test-backend",
            "chain_ids": [],
            "binder_sequences": {},
            "warnings": list(kwargs.get("output_warnings") or []),
        }

    server._run_complex_operation = fake_run_complex_operation
    try:
        result = server.refua_fold(
            name="production_design",
            entities=[{"type": "protein", "id": "A", "sequence": "MKTAYIAK"}],
            feature_output_path="design_features.json",
            feature_output_format="json",
            structure_output_format="mmcif",
            run_boltz=False,
            run_boltzgen=False,
        )
    finally:
        server._run_complex_operation = original

    assert captured["feature_output_format"] == "npz"
    assert str(captured["feature_output_path"]).endswith("design_features.npz")
    assert captured["structure_output_format"] == "cif"
    assert any("normalized to 'npz'" in warning for warning in result.warnings)
    assert any("normalized to 'cif'" in warning for warning in result.warnings)


def test_antibody_design_arg_model_accepts_json_string_compat_inputs() -> None:
    arg_model = server.mcp._tool_manager._tools[
        "refua_antibody_design"
    ].fn_metadata.arg_model
    parsed = arg_model(
        name="schema_probe",
        antibody='{"type":"antibody","ids":["H","L"],"heavy_cdr_lengths":[12,10,14],"light_cdr_lengths":[10,9,9]}',
        context_entities='[{"type":"protein","id":"A","sequence":"MKTAYIAK"}]',
        run_boltz="false",
        run_boltzgen="true",
        return_mmcif=False,
        async_mode=True,
    )
    assert parsed.antibody.startswith("{")
    assert parsed.context_entities.startswith("[")
    assert parsed.run_boltz is False
    assert parsed.run_boltzgen is True


def test_fold_arg_model_accepts_stringified_entity_entries() -> None:
    arg_model = server.mcp._tool_manager._tools["refua_fold"].fn_metadata.arg_model
    parsed = arg_model(
        entities=['{"type":"protein","id":"A","sequence":"MKTAYIAK"}'],
        run_boltz="true",
        run_boltzgen="false",
    )
    assert isinstance(parsed.entities, list)
    assert isinstance(parsed.entities[0], str)
    assert parsed.run_boltz is True
    assert parsed.run_boltzgen is False


def test_non_task_wrapper_returns_structured_content() -> None:
    async def _run() -> None:
        original_get_context = server.mcp.get_context
        try:
            server.mcp.get_context = lambda: SimpleNamespace(
                request_context=SimpleNamespace(experimental=None)
            )
            result = await server._call_tool_with_task_support(
                "refua_validate_spec",
                {
                    "entities": [
                        {"type": "protein", "id": "A", "sequence": "MKTAYIAK"},
                    ]
                },
            )
        finally:
            server.mcp.get_context = original_get_context

        assert isinstance(result, CallToolResult)
        assert isinstance(result.structuredContent, dict)
        assert result.structuredContent.get("valid") is True

    anyio.run(_run)


def test_exploratory_name_guard_blocks_probe_patterns() -> None:
    for name in ("sanity_antibody", "schema_probe", "fold_probe2", "smoke_run"):
        try:
            server._enforce_non_exploratory_execution(
                name,
                allow_exploratory_run=False,
            )
            raise AssertionError("Expected exploratory-name guard to raise.")
        except ValueError as exc:
            assert "blocked by default" in str(exc)


def test_exploratory_name_guard_can_be_overridden() -> None:
    server._enforce_non_exploratory_execution(
        "schema_probe",
        allow_exploratory_run=True,
    )


def test_fold_schema_includes_exploratory_override_flag() -> None:
    tool = next(
        info
        for info in server.mcp._tool_manager.list_tools()
        if info.name == "refua_fold"
    )
    flag = tool.parameters["properties"]["allow_exploratory_run"]
    assert flag["type"] == "boolean"
    assert flag["default"] is False
