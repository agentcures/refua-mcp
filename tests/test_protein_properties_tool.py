from __future__ import annotations

import sys
import types

import pytest

import refua_mcp.server as server


def _install_fake_refua(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("refua")

    class FakeProteinProperties:
        def __init__(self, sequence: str, *, lazy: bool = True, sanitize: bool = True):
            del lazy
            self.sequence = (
                "".join(sequence.split()).upper() if sanitize else str(sequence)
            )

        @classmethod
        def from_sequence(
            cls,
            sequence: str,
            *,
            lazy: bool = True,
            sanitize: bool = True,
        ) -> FakeProteinProperties:
            return cls(sequence, lazy=lazy, sanitize=sanitize)

        def get(self, name: str):
            key = str(name).strip().lower()
            if key == "pi":
                key = "isoelectric_point"
            values = self.to_dict()
            if key not in values:
                raise KeyError(name)
            return values[key]

        def to_dict(self, *, groups=None):
            all_values = {
                "length": len(self.sequence),
                "isoelectric_point": 7.2,
                "gravy": -0.21,
                "net_charge_ph_7_4": 0.5,
            }
            if groups is None:
                return dict(all_values)
            group_members = {
                "basic": {"length", "isoelectric_point", "gravy"},
                "charge": {"isoelectric_point", "net_charge_ph_7_4"},
            }
            selected: set[str] = set()
            for group in groups:
                selected.update(group_members.get(str(group).lower(), set()))
            return {name: all_values[name] for name in all_values if name in selected}

    module.ProteinProperties = FakeProteinProperties
    module.available_protein_properties = lambda: (
        "length",
        "isoelectric_point",
        "gravy",
        "net_charge_ph_7_4",
    )
    module.available_protein_property_groups = lambda: ("basic", "charge")
    monkeypatch.setitem(sys.modules, "refua", module)


def test_refua_protein_properties_tool_is_registered() -> None:
    tool = next(
        info
        for info in server.mcp._tool_manager.list_tools()
        if info.name == "refua_protein_properties"
    )
    assert "sequence" in tool.parameters["properties"]
    assert tool.parameters["properties"]["include_catalog"]["type"] == "boolean"


def test_refua_protein_properties_supports_property_filter_and_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_refua(monkeypatch)
    result = server.refua_protein_properties(
        sequence="mk ta",
        properties='["length", "pi"]',
        include_catalog=True,
    )

    assert result.sequence == "mk ta"
    assert result.normalized_sequence == "MKTA"
    assert result.selected_properties == ["length", "pi"]
    assert result.values == {"length": 4, "pi": 7.2}
    assert result.available_properties == [
        "gravy",
        "isoelectric_point",
        "length",
        "net_charge_ph_7_4",
    ]
    assert result.available_property_groups == ["basic", "charge"]


def test_refua_protein_properties_supports_group_filter_from_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_refua(monkeypatch)
    result = server.refua_protein_properties(
        sequence="MKTAY",
        groups="basic,charge",
    )

    assert result.selected_groups == ["basic", "charge"]
    assert result.values["length"] == 5
    assert result.values["isoelectric_point"] == 7.2
    assert result.values["net_charge_ph_7_4"] == 0.5


def test_refua_protein_properties_rejects_mixed_group_and_property_filters() -> None:
    with pytest.raises(ValueError, match="either properties or groups"):
        server.refua_protein_properties(
            sequence="MKTAY",
            properties=["length"],
            groups=["basic"],
        )
