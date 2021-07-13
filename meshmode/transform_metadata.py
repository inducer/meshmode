"""
.. autoclass:: FirstAxisIsElementsTag
.. autoclass:: ConcurrentElementInameTag
.. autoclass:: ConcurrentDOFInameTag
"""

__copyright__ = """
Copyright (C) 2020-1 University of Illinois Board of Trustees
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

from pytools.tag import Tag


class FirstAxisIsElementsTag(Tag):
    """A tag that is applicable to array outputs indicating that the first
    index corresponds to element indices. This suggests that the implementation
    should set element indices as the outermost loop extent.

    For convenience, this tag may *also* be applied to a kernel if that kernel
    contains exactly one assignment, in which case the tag is considered
    equivalent to being applied to the (single) output array argument.
    """


class ConcurrentElementInameTag(Tag):
    """A tag applicable to an iname indicating that this iname is used to
    iterate over elements in a discretization. States that no dependencies
    exist between elements, i.e. that computations for all elements may be
    performed concurrently.
    """


class ConcurrentDOFInameTag(Tag):
    """A tag applicable to an iname indicating that this iname is used to
    iterate over degrees of freedom (DOFs) within an element in a discretization.
    States that no dependencies exist between output DOFs, i.e. that
    computations for all DOFs within each element may be performed
    concurrently.
    """
