from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
    as gnitstam


def add_tuples(a, b):
    return tuple(ac+bc for ac, bc in zip(a, b))


def tesselateseg():
    node_tuples = [(0,), (1,), (2,)]
    result = [(0, 1), (1, 2)]
    return [node_tuples, result]


def tesselatetri():
    result = []

    node_tuples = list(gnitstam(2, 2))
    node_dict = dict(
          (ituple, idx)
          for idx, ituple in enumerate(node_tuples))

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
    return [node_tuples, result]


def tesselatetet():
    node_tuples = list(gnitstam(2, 3))

    node_dict = dict(
          (ituple, idx)
          for idx, ituple in enumerate(node_tuples))

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

    return [node_tuples, result]
