from __future__ import annotations

import json

import refua_mcp.server as server


def test_refua_fold_schema_uses_discriminated_entities() -> None:
    tool = next(
        info
        for info in server.mcp._tool_manager.list_tools()
        if info.name == "refua_fold"
    )
    entities_items = tool.parameters["properties"]["entities"]["items"]
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
