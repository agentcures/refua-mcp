from __future__ import annotations

import sys
import types

import refua_mcp.server as server


def _install_fake_refua(monkeypatch):
    module = types.ModuleType("refua")

    class FakeProtein:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeDNA:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeRNA:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeBinder:
        def __init__(
            self,
            spec=None,
            length=None,
            template_values=None,
            ids=None,
            binding_types=None,
            secondary_structure=None,
            cyclic=False,
        ):
            self.spec = spec
            self.length = length
            self.template_values = template_values
            self.ids = ids
            self.binding_types = binding_types
            self.secondary_structure = secondary_structure
            self.cyclic = cyclic

    class FakeComplex:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.added = []
            self.file_calls = []

        def add(self, *entities):
            self.added.extend(entities)
            return self

        def file(self, *args, **kwargs):
            self.file_calls.append((args, kwargs))
            return self

    class FakePair:
        def __init__(self, heavy, light):
            self.heavy = heavy
            self.light = light

    class FakeBinderDesigns:
        peptide_calls = []
        disulfide_calls = []
        antibody_calls = []

        @staticmethod
        def peptide(**kwargs):
            FakeBinderDesigns.peptide_calls.append(dict(kwargs))
            return FakeBinder(
                spec=str(kwargs.get("length", 12)),
                ids=kwargs.get("ids", "P"),
                binding_types=kwargs.get("binding_types"),
                secondary_structure=kwargs.get("secondary_structure"),
                cyclic=bool(kwargs.get("cyclic", False)),
            )

        @staticmethod
        def disulfide_peptide(**kwargs):
            FakeBinderDesigns.disulfide_calls.append(dict(kwargs))
            return FakeBinder(
                spec="10C6C3",
                ids=kwargs.get("ids", "P"),
                binding_types=kwargs.get("binding_types"),
                secondary_structure=kwargs.get("secondary_structure"),
                cyclic=bool(kwargs.get("cyclic", True)),
            )

        @staticmethod
        def antibody(**kwargs):
            FakeBinderDesigns.antibody_calls.append(dict(kwargs))
            heavy = FakeBinder(
                spec="heavy-default",
                ids=kwargs.get("heavy_id", "H"),
                binding_types=kwargs.get("heavy_binding_types"),
                secondary_structure=kwargs.get("heavy_secondary_structure"),
                cyclic=bool(kwargs.get("heavy_cyclic", False)),
            )
            light = FakeBinder(
                spec="light-default",
                ids=kwargs.get("light_id", "L"),
                binding_types=kwargs.get("light_binding_types"),
                secondary_structure=kwargs.get("light_secondary_structure"),
                cyclic=bool(kwargs.get("light_cyclic", False)),
            )
            return FakePair(heavy=heavy, light=light)

    class FakeSmallMolecule:
        @staticmethod
        def from_smiles(smiles):
            return f"smiles:{smiles}"

        @staticmethod
        def from_mol(mol, name):
            return f"mol:{name}"

    module.Protein = FakeProtein
    module.DNA = FakeDNA
    module.RNA = FakeRNA
    module.Binder = FakeBinder
    module.BinderDesigns = FakeBinderDesigns
    module.Complex = FakeComplex
    module.SmallMolecule = FakeSmallMolecule
    monkeypatch.setitem(sys.modules, "refua", module)
    return module


def test_binder_template_values_passthrough(monkeypatch):
    fake_refua = _install_fake_refua(monkeypatch)
    complex_spec, _, has_boltz, has_boltzgen = server._build_complex_from_spec(
        name="templated",
        base_dir=None,
        entities=[
            {
                "type": "binder",
                "id": "B",
                "spec": "{core}C{loop}",
                "template_values": {"core": 10, "loop": 6},
            }
        ],
        boltz_mol_dir=None,
    )

    assert has_boltz is False
    assert has_boltzgen is True
    assert len(complex_spec.added) == 1
    binder = complex_spec.added[0]
    assert isinstance(binder, fake_refua.Binder)
    assert binder.template_values == {"core": 10, "loop": 6}
    assert binder.ids == "B"


def test_peptide_disulfide_helper(monkeypatch):
    fake_refua = _install_fake_refua(monkeypatch)
    complex_spec, _, _, has_boltzgen = server._build_complex_from_spec(
        name="peptide",
        base_dir=None,
        entities=[
            {
                "type": "peptide",
                "id": "P",
                "segment_lengths": [8, 5, 4],
                "disulfide": True,
            }
        ],
        boltz_mol_dir=None,
    )

    assert has_boltzgen is True
    assert len(complex_spec.added) == 1
    assert len(fake_refua.BinderDesigns.disulfide_calls) == 1
    call = fake_refua.BinderDesigns.disulfide_calls[0]
    assert call["segment_lengths"] == (8, 5, 4)
    assert call["ids"] == "P"


def test_antibody_helper_with_overrides(monkeypatch):
    fake_refua = _install_fake_refua(monkeypatch)
    complex_spec, _, _, has_boltzgen = server._build_complex_from_spec(
        name="antibody",
        base_dir=None,
        entities=[
            {
                "type": "antibody",
                "ids": ["H", "L"],
                "heavy_cdr_lengths": [12, 10, 14],
                "light_cdr_lengths": "10,9,9",
                "heavy_spec": "{h1}G",
                "heavy_template_values": {"h1": "AA"},
            }
        ],
        boltz_mol_dir=None,
    )

    assert has_boltzgen is True
    assert len(fake_refua.BinderDesigns.antibody_calls) == 1
    call = fake_refua.BinderDesigns.antibody_calls[0]
    assert call["heavy_id"] == "H"
    assert call["light_id"] == "L"
    assert call["heavy_cdr_lengths"] == (12, 10, 14)
    assert call["light_cdr_lengths"] == (10, 9, 9)

    assert len(complex_spec.added) == 2
    heavy, light = complex_spec.added
    assert heavy.spec == "{h1}G"
    assert heavy.template_values == {"h1": "AA"}
    assert light.spec == "light-default"
