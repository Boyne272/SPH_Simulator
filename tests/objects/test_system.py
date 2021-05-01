"""Unit tests for system.py."""

import json

from pytest import raises

from sph.objects.system import System


def test_reload_seed():
    """Ensure a reloaded system is identical (with seeded generator)."""
    system = System()
    sys_data = system.as_json(rand=False)
    re_sys_dict = json.loads(sys_data)

    re_system = System.from_dict(re_sys_dict)

    assert re_system.as_json(rand=False) == sys_data
    assert '%r' % re_system.rand == '%r' % system.rand
    assert re_system.summary == system.summary
    assert re_system == system


def test_reload_rand():
    """Ensure a reloaded system is identical (with full random generator)."""
    system = System()
    sys_data = system.as_json()
    re_sys_dict = json.loads(sys_data)

    re_system = System.from_dict(re_sys_dict)

    assert re_system.as_json() == sys_data
    assert '%r' % re_system.rand == '%r' % system.rand
    assert re_system.summary == system.summary
    assert re_system == system


def test_reload_failure():
    """Ensure type casing throws expected errors."""
    with raises(ValueError):
        System.from_dict({'h_fac': 'i am not a float'})
