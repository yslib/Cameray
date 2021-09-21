from typing import List
import numpy as np
from .imgio import Image, ImageBracket

class Param:
    def __init__(self, name, set_cb=None):
        self._name = name
        self._set_cb = set_cb

    def _set_val(self, ins, val):
        ins.__dict__[self._name] = val
        callable(self._set_cb) and self._set_cb(val)

    def __get__(self, ins, cls):
        if ins is None:
            return self
        else:
            return ins.__dict__[self._name]

    def __delete__(self, ins):
        del ins.__dict__[self._name]

class IntParam(Param):
    def __init__(self, name, min:int=0,max:int=100, cb=None):
        super(IntParam, self).__init__(name, cb)
        self.min_value = min
        self.max_value = max

    def __set__(self, ins, val):
        if not isinstance(val, int):
            raise TypeError('Expected an int')
        val = max(self.min_value, min(val,self.max_value))
        self._set_val(ins, val)


class FloatParam(Param):
    def __init__(self, name:str, min:float=0.0, max:float=1.0, cb=None):
        super(FloatParam, self).__init__(name,cb)
        self.min_value = min
        self.max_value = max

    def __set__(self, ins, val):
        if not isinstance(val, float):
            raise TypeError('Expected float type')
        val = max(self.min_value, min(val,self.max_value))
        self._set_val(ins, val)


class Vec3Param(Param):
    def __init__(self, name,cb=None):
        super(IntParam, self).__init__(name, cb)

    def __set__(self, ins, val):
        if not isinstance(val, tuple):
            raise TypeError('Expected tuple')
        self._set_val(ins, val)


class ImageParam(Param):
    def __init__(self, name,val, cb=None):
        super(ImageParam, self).__init__(name,val,cb)

    def __set__(self, ins, val):
        if not isinstance(val, Image):
            raise TypeError('Expected tuple')
        self._set_val(ins, val)


class ImageBracketParam:
    def __init__(self, name, cb):
        super(ImageBracketParam, self).__init__(name, cb)

    def __set__(self, ins, val):
        if not isinstance(val, ImageBracket):
            raise TypeError('Expected tuple')
        self._set_val(ins, val)
