from refua_mcp.server import _parquet_columns_for_query


def test_parquet_read_includes_filter_columns_omitted_from_projection() -> None:
    columns = _parquet_columns_for_query(["smiles"], {"activity": {"gt": 0.5}})
    assert columns == ["smiles", "activity"]


def test_parquet_read_keeps_unrestricted_projection() -> None:
    assert _parquet_columns_for_query(None, {"activity": {"gt": 0.5}}) is None
    assert _parquet_columns_for_query(["smiles"], {}) == ["smiles"]
