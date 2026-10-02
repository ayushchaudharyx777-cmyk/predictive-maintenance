from validate import validate


def test_clean_data_passes(raw):
    assert validate(raw) == []


def test_gap_is_reported(raw):
    bad = dict(raw, telemetry=raw["telemetry"].drop(index=100))
    assert any("gaps" in p for p in validate(bad))


def test_negative_sensor_and_unknown_machine(raw):
    tel = raw["telemetry"].copy()
    tel.loc[0, "volt"] = -1
    errors = raw["errors"].copy()
    errors.loc[0, "machineID"] = 9999
    problems = validate(dict(raw, telemetry=tel, errors=errors))
    assert any("negative" in p for p in problems) and any("9999" in p for p in problems)


def test_missing_column(raw):
    bad = dict(raw, machines=raw["machines"].drop(columns="age"))
    assert any("missing columns" in p for p in validate(bad))
