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
