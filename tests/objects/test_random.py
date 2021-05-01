"""Unit tests for random.py."""

from numpy import testing

from sph.objects import RngState

def test_rng_save():
    """Ensure a random state can be saved as string and loaded to the give the same values."""
    rand1 = RngState()
    rand2 = RngState.from_string(rand1.as_string())

    assert str(rand1.get_state()) == str(rand2.get_state()), 'the states should be identical'
    assert '%r' % rand1 == '%r' % rand2, 'the md5 hashes should be the same'

    testing.assert_array_equal(rand1.random(1000), rand2.random(1000))
    testing.assert_array_equal(rand1.normal(1000), rand2.normal(1000))
    testing.assert_array_equal(rand1.binomial(1000, 0.5), rand2.binomial(1000, 0.5))
