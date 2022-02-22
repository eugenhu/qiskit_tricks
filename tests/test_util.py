import numpy as np

from qiskit_tricks.util import (
    bit_extract,
    ensure_int_state,
    get_play_instruction,
    marginal_counts,
)


def test_marginal_counts_int():
    counts = {
        0b0000: 10,
        0b0001:  7,
        0b0010:  3,
        0b0100:  5,
        0b1000:  2,
    }

    subsystem_counts = marginal_counts(counts, subsystem=[{2, 3}, {0}, {99}])

    assert subsystem_counts \
        == [{0: 20,
             1:  5,
             2:  2},
            {0: 20,
             1:  7},
            {0: 27}]


def test_marginal_counts_str_bin():
    counts = {
        '0000': 10,
        '0001':  7,
        '0010':  3,
        '0100':  5,
        '1000':  2,
    }

    subsystem_counts = marginal_counts(counts, subsystem=[{2, 3}, {0}])

    assert subsystem_counts \
        == [{0: 20,
             1:  5,
             2:  2},
            {0: 20,
             1:  7}]


def test_marginal_counts_str_hex():
    counts = {
        hex(0b0000): 10,
        hex(0b0001):  7,
        hex(0b0010):  3,
        hex(0b0100):  5,
        hex(0b1000):  2,
    }

    subsystem_counts = marginal_counts(counts, subsystem=[{2, 3}, {0}])

    assert subsystem_counts \
        == [{0: 20,
             1:  5,
             2:  2},
            {0: 20,
             1:  7}]


def test_ensure_int_state():
    state = 123

    got = ensure_int_state(state)
    assert got == state and isinstance(got, int)

    got = ensure_int_state(bin(state))
    assert got == state and isinstance(got, int)

    # Strip leading 0b
    got = ensure_int_state(bin(state)[2:])
    assert got == state and isinstance(got, int)

    got = ensure_int_state(hex(state))
    assert got == state and isinstance(got, int)

    import numpy as np
    got = ensure_int_state(np.int8(state))
    assert got == state and isinstance(got, int)


def test_bit_extract():
    assert bit_extract(0b11010001, [0, 4, 1, 1]) == 0b0011


def test_bit_extract_numpy():
    source = np.array([
        0b11010001,
        0b00101110,
    ])

    assert np.all(bit_extract(source, [0, 4, 1, 1]) == np.array([0b0011, 0b1100]))


def test_get_play_instruction():
    from qiskit.pulse import Schedule, ShiftPhase, Gaussian, Play, DriveChannel

    d0 = DriveChannel(0)
    play = Play(Gaussian(160, 0.2, 40), d0)

    schedule = Schedule(
        ShiftPhase(3.14, d0),
        play,
        ShiftPhase(-3.14, d0),
    )

    assert get_play_instruction(schedule) == play
