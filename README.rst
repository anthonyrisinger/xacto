Xacto
~~~~~

**a typelib and CLI analyzer/generator**

Introspect, compose, marshal and export arbitrary callables into a unified,
hierarchical, command-line interface (CLI)

Features
~~~~~~~~

auto-find tools
scan signatures and export as CLI node
if default is callable, call it to setup arg

Why
---

FAST! EASY! natural import usage! --help is decent-ish with types/etc! AH!

Quickstart
----------

#. Install::

    # pip install xacto

#. Run::

    # ln -s /path/to/bin/xacto do

#. Create python file at ``tools/work.py``::

    __all__ = ["easy", "hard"]
    #__xacto__ = {...}

    def easy(method, speed=16, *tasks):
        print(locals())
    #easy.__xacto__ = {...}

    def hard(method, speed=32, *tasks, **params):
        print(locals())
    #hard.__xacto__ = {...}

#. Run::

    # ./do work easy --method=never task1 task2

#. Run::

    # ./do work hard --speed=64 --method=cheat --code=iddqd taskN

Limitations
-----------

- true/false quirkyness (default=True means --default flips to False)
- can't handle lists (--arg a b c --other ...)

TODO
----

- RELEASE!
- setuptools
- symlink to binary to create new
- testing: set-like functions, import semantics.. everything
- handle bools better
- handle lists (--arg a b c --other ...)
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
- bind tools to xacto object (eg. celery/waf)
