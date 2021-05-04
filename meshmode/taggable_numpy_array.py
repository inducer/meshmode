import numpy as np
from pytools.tag import Taggable, Tag

class TaggableNumpyArray(np.ndarray, Taggable):

    def __init__(self, *args, tags=frozenset(), **kwargs):
        Taggable.__init__(self, tags=tags)

    def __new__(cls, input_array, tags=frozenset()):

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the new attribute to the created instance
        # Apparently this Taggable __init__ function is called so this
        # is not necessary.
        #obj.tags = tags

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'tags', None)
