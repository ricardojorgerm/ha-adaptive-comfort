"""House-other temperature helper for inter-zone mixing."""

from custom_components.adaptive_comfort.core.thermal import house_other_temperature


def test_house_other_excludes_self():
    readings = {
        "a": (22.0, 30.0),
        "b": (20.0, 40.0),
    }
    assert house_other_temperature("a", readings) == 20.0
    assert house_other_temperature("b", readings) == 22.0


def test_house_other_includes_aux():
    readings = {"a": (24.0, 30.0)}
    aux = [(21.0, 10.0)]
    # b excluded: only aux contributes -> 21
    assert house_other_temperature("a", readings, aux) == 21.0


def test_house_other_none_when_solo():
    assert house_other_temperature("only", {"only": (22.0, 30.0)}) is None
