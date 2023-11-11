'''
@author: Steven Tang <steven.tang@bytedance.com>
'''
from ._testextensionmodule import callNativeFuncA, callNativeFuncAThroughDlSym

__all__ = ("callNativeFuncA", "callNativeFuncAThroughDlSym")