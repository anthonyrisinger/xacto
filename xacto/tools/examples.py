# encoding: UTF-8
"""
Collection of examples

These tools perform zero useful work!
"""


def echo(a0, a1=1, a2=False, a3='str', *args, **kwds):
    """
    Simple echo tool

    Pretty print everything!
    """
    __import__('pprint').pprint(locals())


class echo(object):
    """
    doc0 (echo)

    doc1 (echo)
    """

    def arg0(self, item0=0, **kwds):
        """docstring (arg0)"""

    def arg1(self, one, two, **kwds):
        """docstring (arg1)"""

    def arg2(self, item0, item1, *items, **kwds):
        """docstring (arg2)"""

    def args(self, **kwds):
        """docstring (args)"""

    def kwds(self, **kwds):
        """docstring (kwds)"""

    def __call__(self, arg0, arg1=1, arg2=False, arg3='str', *args, **kwds):
        """
        Simple echo tool

        Pretty print everything!
        """
        __import__('pprint').pprint(locals())
