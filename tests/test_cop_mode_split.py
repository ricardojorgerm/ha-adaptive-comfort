"""Mode-split COP ledger helpers."""

from custom_components.adaptive_comfort.core import power


def test_migrate_heads_and_banded_to_cool_prefix():
    heads = power.migrate_cop_table_keys({"1": [1.2, 10], "2": [2.0, 5]}, kind="heads")
    assert heads == {"cool|1": (1.2, 10), "cool|2": (2.0, 5)}
    banded = power.migrate_cop_table_keys({"1|hot": [0.9, 20], "3|mild": [2.4, 40]}, kind="banded")
    assert banded["cool|1|hot"] == (0.9, 20)
    assert banded["cool|3|mild"] == (2.4, 40)


def test_migrate_state_and_passthrough_prefixed():
    state = power.migrate_cop_table_keys(
        {"park": [3.3, 44], "cool|conditioning": [0.8, 10]}, kind="state"
    )
    assert state["cool|park"] == (3.3, 44)
    assert state["cool|conditioning"] == (0.8, 10)
    # heat prefix must not be rewritten
    heat = power.migrate_cop_table_keys({"heat|2": [2.5, 8]}, kind="heads")
    assert heat == {"heat|2": (2.5, 8)}


def test_cop_by_head_count_filters_mode():
    table = {
        "cool|1": (1.0, 30),
        "cool|2": (1.5, 25),
        "heat|1": (2.5, 30),
        "cool|1|hot": (0.9, 40),  # banded key must not leak into heads map
    }
    cool = power.cop_by_head_count(table, "cool", min_samples=20)
    heat = power.cop_by_head_count(table, "heat", min_samples=20)
    assert cool == {1: 1.0, 2: 1.5}
    assert heat == {1: 2.5}


def test_cop_by_band_aggregates_within_mode_only():
    table = {
        "cool|1|hot": (0.9, 20),
        "cool|3|hot": (1.1, 20),
        "cool|3|mild": (2.4, 40),
        "heat|1|mild": (3.0, 40),
    }
    cool = power.cop_by_band(table, "cool", min_samples=20)
    heat = power.cop_by_band(table, "heat", min_samples=20)
    assert abs(cool["hot"] - 1.0) < 1e-9  # mean of 0.9 and 1.1
    assert abs(cool["mild"] - 2.4) < 1e-9
    assert "mild" in heat and abs(heat["mild"] - 3.0) < 1e-9
    assert "hot" not in heat


def test_cop_sample_mode_requires_controller_and_plant_agree():
    assert (
        power.cop_sample_mode("cool", any_heating=False, any_cooling=True) == "cool"
    )
    assert (
        power.cop_sample_mode("heat", any_heating=True, any_cooling=False) == "heat"
    )
    # Controller heat but heads still reporting cool (lag) → skip.
    assert power.cop_sample_mode("heat", any_heating=False, any_cooling=True) is None
    # Controller cool, no cooling heads → skip (do not default to cool).
    assert power.cop_sample_mode("cool", any_heating=False, any_cooling=False) is None
    # Idle / auto controller → never file.
    assert power.cop_sample_mode("off", any_heating=False, any_cooling=True) is None
    assert power.cop_sample_mode("auto", any_heating=True, any_cooling=False) is None
