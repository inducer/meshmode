__copyright__ = """
Copyright (C) 2018 Andreas Kloeckner
Copyright (C) 2014-6 Shivam Gupta
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
    as gnitstam


def mul_tuples(t, m):
    return tuple(m * a for a in t)


def add_tuples(a, b):
    return tuple(ac+bc for ac, bc in zip(a, b))


def halve_tuple(a):
    def halve(x):
        d, r = divmod(x, 2)
        if r:
            raise ValueError("%s is not evenly divisible by two" % x)
        return d

    return tuple(halve(ac) for ac in a)


def tesselateseg():
    node_tuples = [(0,), (1,), (2,)]
    result = [(0, 1), (1, 2)]
    return node_tuples, result


def tesselatetri():
    result = []

    node_tuples = list(gnitstam(2, 2))
    node_dict = {
          ituple: idx
          for idx, ituple in enumerate(node_tuples)}

    def try_add_tri(current, d1, d2, d3):
        try:
            result.append((
                node_dict[add_tuples(current, d1)],
                node_dict[add_tuples(current, d2)],
                node_dict[add_tuples(current, d3)],
                ))
        except KeyError:
            pass

    if len(result) > 0:
        return [node_tuples, result]
    for current in node_tuples:
        # this is a tesselation of a square into two triangles.
        # subtriangles that fall outside of the master tet are simply not added.

        # positively oriented
        try_add_tri(current, (0, 0), (1, 0), (0, 1))
        try_add_tri(current, (1, 0), (1, 1), (0, 1))
    return node_tuples, result


def tesselatetet():
    node_tuples = list(gnitstam(2, 3))

    node_dict = {
          ituple: idx
          for idx, ituple in enumerate(node_tuples)}

    def try_add_tet(current, d1, d2, d3, d4):
        try:
            result.append((
                node_dict[add_tuples(current, d1)],
                node_dict[add_tuples(current, d2)],
                node_dict[add_tuples(current, d3)],
                node_dict[add_tuples(current, d4)],
                ))
        except KeyError:
            pass

    result = []

    if len(result) > 0:
        return [node_tuples, result]
    for current in node_tuples:
        # this is a tesselation of a cube into six tets.
        # subtets that fall outside of the master tet are simply not added.

        # positively oriented
        try_add_tet(current, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        try_add_tet(current, (1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0))
        try_add_tet(current, (1, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 1))

        try_add_tet(current, (1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0))
        try_add_tet(current, (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1))
        try_add_tet(current, (0, 1, 1), (1, 1, 1), (1, 0, 1), (1, 1, 0))

    return node_tuples, result


def tesselate_simplex_bisection(dim):
    if dim == 1:
        return tesselateseg()
    elif dim == 2:
        return tesselatetri()
    elif dim == 3:
        return tesselatetet()
    else:
        raise ValueError("cannot tesselate %d-simplex" % dim)
