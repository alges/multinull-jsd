"""
Tests for the multinull_jsd package's __init__.py module.
"""
import multinull_jsd as pkg


def test_available_cdf_backends_includes_expected() -> None:
    """
    Test that the available_cdf_backends function returns expected backend names.
    """
    backend_names: tuple[str, ...] = pkg.available_cdf_backends()
    expected_backends = {"exact", "mc_multinomial", "mc_normal"}
    assert expected_backends.issubset(backend_names)
    assert isinstance(backend_names, tuple)
    assert list(backend_names) == sorted(backend_names)


def test_lazy_getattr_is_exposed() -> None:
    """
    Test that the lazy __getattr__ correctly exposes the MultiNullJSDTest class.
    """
    cls: pkg.MultiNullJSDTest = getattr(pkg, "MultiNullJSDTest")
    assert isinstance(cls, type)
    assert cls.__name__ == "MultiNullJSDTest"


def test_dunder_dir_contains_public_api() -> None:
    """
    Test that the __dir__ function includes the expected public API elements.
    """
    names: list[str] = dir(pkg)
    assert "MultiNullJSDTest" in names
    assert "available_cdf_backends" in names


def test_unknown_attribute_raises_attribute_error() -> None:
    """
    Test that accessing a nonexistent attribute raises AttributeError with the correct message.
    """
    try:
        getattr(pkg, "NonexistentAttribute")
    except AttributeError as e:
        exception_msg: str = str(e)
        assert "multinull_jsd" in exception_msg and "NonexistentAttribute" in exception_msg
    else:
        raise AssertionError("Expected AttributeError was not raised")
