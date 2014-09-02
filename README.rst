Xacto
~~~~~

**CLI Analyzer/Generator**

Introspect, compose, marshal and export arbitrary callables into a unified,
hierarchical, command-line interface (CLI)

Features
~~~~~~~~

auto-find tools
scan signatures and export as CLI node

Why
---

FAST! EASY! natural import usage! ``--help`` is decent!

Quickstart
----------

#. Install::

    # pip install xacto

#. Prepare::

    # ln -s $(which xacto) do

#. Create python file at ``tools/work.py``::

    from pprint import pprint


    __all__ = ["easy", "hard", "manual"]


    def easy(method, speed=16, *tasks):
        """The simple version"""
        pprint(locals())


    def hard(method, speed=32, *tasks, **params):
        """The difficult version"""
        pprint(locals())


    class manual(object):
        """The laborious version"""

        def __call__(self, method, speed, *tasks):
            pprint(locals())

        def method(self, process):
            """Howto perform operation"""

        def speed(self, ops=64):
            """Operations per second"""

        def tasks(self, pri, sec, ter):
            """Various tasks"""

#. View ``--help``::

    # ./do work --help
    usage: do work OBJECT ...

    additional modules:
      OBJECT  := { .do.work }
        easy  The simple version
        hard  The difficult version
        manual
              The laborious version

#. View object-level ``--help``::

    # ./do work manual --help
    usage: do work manual --method PROCESS --speed [OPS] [tasks [tasks ...]]

    The laborious version

    positional arguments:
      tasks             Various tasks

    optional arguments:
      --method PROCESS  Howto perform operation
      --speed [OPS]     Operations per second

    The laborious version

#. Run tool (function)::

    # ./do work hard --method=cheat --code=iddqd taskN
    {'method': 'cheat',
     'params': {'code': 'iddqd'},
     'speed': 32,
     'tasks': ('taskN',)}

#. Run tool (class)::

    # ./do work manual --method=cheap --speed=256 taskN
    {'method': 'cheap',
     'self': <do.work.manual object at ...>,
     'speed': '256',
     'tasks': ('taskN',)}

Limitations
-----------

- true/false quirkyness (default=True means --default flips to False)

TODO
----

- RELEASE!
- testing: set-like functions, import semantics.. everything
- handle bools better
- detect output
- standard output structure
- prettify to tty
- lazy load tools
- lazy import globals (cpython)
- bytecode cache
- argument forwarding/chaining
- integrate with zippy.shell
- tab-completion
- auto-reduce common components for aliases
- make xacto object accessible to tools
