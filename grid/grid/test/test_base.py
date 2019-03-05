# -*- coding: utf-8 -*-
# OLDGRIDS: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The OLDGRIDS Development Team
#
# This file is part of OLDGRIDS.
#
# OLDGRIDS is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# OLDGRIDS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import numpy as np

from grid import *  # pylint: disable=wildcard-import,unused-wildcard-import


def test_grid_integrate():
    npoint = 10
    grid = IntGrid(np.random.normal(0, 1, (npoint, 3)), np.random.normal(0, 1, npoint))
    pot = np.random.normal(0, 1, npoint)
    dens = np.random.normal(0, 1, npoint)

    # three
    int1 = grid.integrate(pot, dens)
    int2 = (grid.weights * pot * dens).sum()
    assert abs(int1 - int2) < 1e-10

    # two
    int1 = grid.integrate(pot)
    int2 = (grid.weights * pot).sum()
    assert abs(int1 - int2) < 1e-10

    # one
    int1 = grid.integrate()
    int2 = grid.weights.sum()
    assert abs(int1 - int2) < 1e-10


def test_grid_integrate_segments():
    npoint = 10
    segments = np.array([2, 5, 3])
    grid = IntGrid(np.random.normal(0, 1, (npoint, 3)), np.random.normal(0, 1, npoint))
    pot = np.random.normal(0, 1, npoint)
    dens = np.random.normal(0, 1, npoint)

    # three
    ints = grid.integrate(pot, dens, segments=segments)
    assert ints.shape == (3,)
    product = grid.weights * pot * dens
    assert abs(ints[0] - product[:2].sum()) < 1e-10
    assert abs(ints[1] - product[2:7].sum()) < 1e-10
    assert abs(ints[2] - product[7:].sum()) < 1e-10

    # two
    ints = grid.integrate(pot, segments=segments)
    assert ints.shape == (3,)
    product = grid.weights * pot
    assert abs(ints[0] - product[:2].sum()) < 1e-10
    assert abs(ints[1] - product[2:7].sum()) < 1e-10
    assert abs(ints[2] - product[7:].sum()) < 1e-10

    # one
    ints = grid.integrate(segments=segments)
    assert ints.shape == (3,)
    assert abs(ints[0] - grid.weights[:2].sum()) < 1e-10
    assert abs(ints[1] - grid.weights[2:7].sum()) < 1e-10
    assert abs(ints[2] - grid.weights[7:].sum()) < 1e-10


def test_grid_integrate_cartesian_moments():
    npoint = 10
    grid = IntGrid(np.random.normal(0, 1, (npoint, 3)), np.random.normal(0, 1, npoint))
    dens = np.random.normal(0, 1, npoint)

    center = np.random.normal(0, 1, 3)
    x = grid.points[:, 0] - center[0]
    y = grid.points[:, 1] - center[1]
    z = grid.points[:, 2] - center[2]

    ints = grid.integrate(dens, center=center, lmax=2, mtype=1)
    assert ints.shape == (10,)
    assert abs(ints[0] - (grid.weights * dens).sum()) < 1e-10
    assert abs(ints[1] - (grid.weights * dens * x).sum()) < 1e-10
    assert abs(ints[2] - (grid.weights * dens * y).sum()) < 1e-10
    assert abs(ints[3] - (grid.weights * dens * z).sum()) < 1e-10
    assert abs(ints[4] - (grid.weights * dens * x * x).sum()) < 1e-10
    assert abs(ints[5] - (grid.weights * dens * x * y).sum()) < 1e-10
    assert abs(ints[9] - (grid.weights * dens * z * z).sum()) < 1e-10


def test_grid_integrate_cartesian_moments_segments():
    npoint = 10
    segments = np.array([2, 5, 3])
    grid = IntGrid(np.random.normal(0, 1, (npoint, 3)), np.random.normal(0, 1, npoint))
    dens = np.random.normal(0, 1, npoint)

    center = np.random.normal(0, 1, 3)
    x = grid.points[:, 0] - center[0]
    y = grid.points[:, 1] - center[1]
    z = grid.points[:, 2] - center[2]

    ints = grid.integrate(dens, center=center, lmax=2, mtype=1, segments=segments)
    assert ints.shape == (3, 10)
    for i, begin, end in (0, 0, 2), (1, 2, 7), (2, 7, 10):
        assert abs(ints[i, 0] - (grid.weights * dens)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 1] - (grid.weights * dens * x)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 2] - (grid.weights * dens * y)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 3] - (grid.weights * dens * z)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 4] - (grid.weights * dens * x * x)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 5] - (grid.weights * dens * x * y)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 9] - (grid.weights * dens * z * z)[begin:end].sum()) < 1e-10


def test_grid_integrate_pure_moments():
    npoint = 10
    grid = IntGrid(np.random.normal(0, 1, (npoint, 3)), np.random.normal(0, 1, npoint))
    dens = np.random.normal(0, 1, npoint)

    center = np.random.normal(0, 1, 3)
    x = grid.points[:, 0] - center[0]
    y = grid.points[:, 1] - center[1]
    z = grid.points[:, 2] - center[2]
    r2 = x * x + y * y + z * z

    ints = grid.integrate(dens, center=center, lmax=2, mtype=2)
    assert ints.shape == (9,)
    assert abs(ints[0] - (grid.weights * dens).sum()) < 1e-10
    assert abs(ints[1] - (grid.weights * dens * z).sum()) < 1e-10
    assert abs(ints[2] - (grid.weights * dens * x).sum()) < 1e-10
    assert abs(ints[3] - (grid.weights * dens * y).sum()) < 1e-10
    assert abs(ints[4] - (grid.weights * dens * (1.5 * z ** 2 - 0.5 * r2)).sum()) < 1e-10
    assert abs(ints[5] - (grid.weights * dens * (3.0 ** 0.5 * x * z)).sum()) < 1e-10
    assert abs(ints[8] - (grid.weights * dens * (3.0 ** 0.5 * x * y)).sum()) < 1e-10


def test_grid_integrate_pure_moments_segments():
    npoint = 10
    segments = np.array([2, 5, 3])
    grid = IntGrid(np.random.normal(0, 1, (npoint, 3)), np.random.normal(0, 1, npoint))
    dens = np.random.normal(0, 1, npoint)

    center = np.random.normal(0, 1, 3)
    x = grid.points[:, 0] - center[0]
    y = grid.points[:, 1] - center[1]
    z = grid.points[:, 2] - center[2]
    r2 = x * x + y * y + z * z

    ints = grid.integrate(dens, center=center, lmax=2, mtype=2, segments=segments)
    assert ints.shape == (3, 9)
    for i, begin, end in (0, 0, 2), (1, 2, 7), (2, 7, 10):
        assert abs(ints[i, 0] - (grid.weights * dens)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 1] - (grid.weights * dens * z)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 2] - (grid.weights * dens * x)[begin:end].sum()) < 1e-10
        assert abs(ints[i, 3] - (grid.weights * dens * y)[begin:end].sum()) < 1e-10
        assert abs(
            ints[i, 4] - (grid.weights * dens * (1.5 * z ** 2 - 0.5 * r2))[begin:end].sum()) < 1e-10
        assert abs(
            ints[i, 5] - (grid.weights * dens * (3.0 ** 0.5 * x * z))[begin:end].sum()) < 1e-10
        assert abs(
            ints[i, 8] - (grid.weights * dens * (3.0 ** 0.5 * x * y))[begin:end].sum()) < 1e-10


def test_grid_integrate_radial_moments():
    npoint = 10
    grid = IntGrid(np.random.normal(0, 1, (npoint, 3)), np.random.normal(0, 1, npoint))
    dens = np.random.normal(0, 1, npoint)

    center = np.random.normal(0, 1, 3)
    x = grid.points[:, 0] - center[0]
    y = grid.points[:, 1] - center[1]
    z = grid.points[:, 2] - center[2]
    r = np.sqrt(x * x + y * y + z * z)

    ints = grid.integrate(dens, center=center, lmax=2, mtype=3)
    assert abs(ints[0] - (grid.weights * dens).sum()) < 1e-10
    assert abs(ints[1] - (grid.weights * dens * r).sum()) < 1e-10
    assert abs(ints[2] - (grid.weights * dens * r * r).sum()) < 1e-10


def test_dot_multi():
    npoint = 10
    pot = np.random.normal(0, 1, npoint)
    dens = np.random.normal(0, 1, npoint)

    # two
    dot1 = np.dot(pot, dens)
    dot2 = dot_multi(pot, dens)
    assert abs(dot1 - dot2) < 1e-10

    # one
    dot1 = pot.sum()
    dot2 = dot_multi(pot)
    assert abs(dot1 - dot2) < 1e-10
