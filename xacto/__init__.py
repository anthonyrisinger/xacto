#!/opt/advent/bin/python2.7
# coding: utf-8
"""
Xacto CLI

High-level operational tool encapsulating all others!
"""


from __future__ import absolute_import
from __future__ import print_function


import sys, os
import warnings
from os import path as pth

import argparse, pkgutil, types, inspect, imp, re
from itertools import imap, izip, ifilter, repeat, chain


__version__ = '0.6.2'


nil = type('nil', (int,), dict())(0)


class mapo(dict):

    __slots__ = (
        '__dict__',
        )

    __types__ = dict()
    __feature__ = None

    def __copy__(self):
        return self.__class__(self)
    copy = __copy__

    @classmethod
    def type(cls, key=nil, attr=nil):
        if key is nil:
            return cls.__types__

        if attr is not nil:
            cls.__types__[key] = attr

        return cls.__types__.get(key, nil)

    @classmethod
    def features(cls, sep=None):
        feats = list()
        for base in reversed(cls.__mro__):
            feature = getattr(base, '__feature__', None)
            if feature:
                feats.append(feature)
        feats = tuple(feats)
        if hasattr(sep, 'join'):
            feats = sep.join(feats)
        return feats

    @classmethod
    def feature(cls, fun=None, *args, **kwds):
        def g(fun):
            key = kwds.get('key')
            merge = (not hasattr(fun, '__mro__')) and {fun.__name__: fun}
            if not key and not merge:
                key = fun.__name__
            if not key:
                raise TypeError('error: cannot derive key: %s' % (fun,))

            if merge:
                fun = cls.type(key) or dict()
                fun.update(merge)
            cls.type(key, fun)
            return fun if merge is None else None

        return g if fun is None else g(fun=fun)

    @classmethod
    def matic(cls, *args, **kwds):
        #TODO: should features express ordering constraints?
        feats = kwds.get('features') or tuple()
        if feats[0:0] == '':
            feats = tuple(feats.split())

        bases = list()
        for key in reversed(feats):
            typ = cls.type(key)
            if not typ:
                raise TypeError('error: undefined feature: %s' % (key,))

            typ = typ if hasattr(typ, '__mro__') else type('x', (cls,), typ)
            typ.__feature__ = key
            typ.__name__ = typ.features(sep='_')
            cls.type(key, typ)
            bases.append(typ)

        #TODO: should this be cached?
        typ = type('x', tuple(bases + [cls]), dict())
        typ.__feature__ = None
        typ.__name__ = typ.features(sep='_')
        return typ

@mapo.feature(key='attr')
class feature(mapo):
    def __new__(cls, *args, **kwds):
        supr = super(cls.type('attr'), cls)
        self = self.__dict__ = supr.__new__(cls, *args, **kwds)
        return self

@mapo.feature(key='autoa')
class feature(mapo):
    def __getattr__(self, key):
        supr = super(self.type('autoa'), self)
        try:
            return supr.__getattr__(key)
        except (KeyError, AttributeError):
            attr = self[key] = self.__class__()
            return attr

@mapo.feature(key='autoi')
class feature(mapo):
    def __missing__(self, key):
        supr = super(self.type('autoi'), self)
        try:
            return supr.__missing__(key)
        except (KeyError, AttributeError):
            attr = self[key] = self.__class__()
            return attr

@mapo.feature(key='auto')
class feature(mapo.type('autoa'), mapo.type('autoi')):
    pass

@mapo.feature(key='set')
class feature(mapo):
    __or__   = lambda s, o: s.__oper__(o, 'or')
    __xor__  = lambda s, o: s.__oper__(o, 'xor')
    __and__  = lambda s, o: s.__oper__(o, 'and')
    __sub__  = lambda s, o: s.__oper__(o, 'sub')
    __ror__  = lambda s, o: s.__oper__(o, 'ror')
    __rxor__ = lambda s, o: s.__oper__(o, 'rxor')
    __rand__ = lambda s, o: s.__oper__(o, 'rand')
    __rsub__ = lambda s, o: s.__oper__(o, 'rsub')
    __ior__  = lambda s, o: s.__oper__(o, 'ior')
    __ixor__ = lambda s, o: s.__oper__(o, 'ixor')
    __iand__ = lambda s, o: s.__oper__(o, 'iand')
    __isub__ = lambda s, o: s.__oper__(o, 'isub')
    def __oper__(self, other, op):
        iop = (op[0]=='i')
        rop = (op[0]=='r')
        typ = self.__class__
        get = lambda x: None
        if hasattr(other, 'keys'):
            get = other.__getitem__
        if rop:
            self, other = typ((x, get(x)) for x in other), self
            get = other.__getitem__
        elif not iop:
            self = self.copy()

        try:
            ikeys = self.viewkeys()
        except AttributeError:
            ikeys = frozenset(self)
        try:
            okeys = other.viewkeys()
        except AttributeError:
            okeys = frozenset(other)

        op = '__%s__' % op.lstrip('ir')
        oper = getattr(ikeys, op)
        keys = oper(okeys)
        if keys is NotImplemented:
            keys = oper(frozenset(okeys))
        if keys is NotImplemented:
            return keys

        get = lambda x: None
        if hasattr(other, 'keys'):
            get = other.__getitem__
        for key in keys - ikeys:
            self[key] = get(key)
        for key in ikeys - keys:
            del self[key]

        return self


record = mapo.matic(features='attr set')
automap = record.matic(features='auto')


class Namespace(object):

    def __init__(self, ns, hint=None, parent=None, **kwds):
        self._ns = ns
        self._parent = parent
        self._parser = None
        self._subparser = None
        self._module = None
        self._code = None
        self._path = None
        self._keys = kwds
        if hasattr(self._parent, 'pathspec'):
            package = self._parent._module
            if package:
                package.__package__ = str(self._parent)
                package.__path__ = getattr(package, '__path__', None) or list()
                if self._parent.path not in package.__path__:
                    package.__path__.append(self._parent.path)
        else:
            self._parent = None
            self._cache = dict()
            if pth.exists(parent):
                parent = pth.abspath(parent)
            self._path = parent

        docstr = pathname = None
        if isinstance(hint, types.ModuleType):
            module = self._module = hint
            pathname = module.__file__
            self._code = True
            if module.__name__ == '__main__':
                module.__name__ = str(self)
            if getattr(module, '__doc__', None):
                docstr = module.__doc__.strip()
        else:
            module = self._module = types.ModuleType(str(self))
            module.__package__ = module.__name__
            module.__file__ = None
            if isinstance(hint, types.CodeType):
                code = self._code = hint
                pathname = code.co_filename
            elif hint is None:
                pathname = self.path
            else:
                pathname = str(hint)

        pkgpath = list()
        if self._path:
            pkgpath.append(self._path)

        if pth.exists(pathname):
            pathname = pth.abspath(pathname)
        if pth.isfile(pathname):
            module.__file__ = pathname
        elif pth.isdir(pathname) or not pathname.endswith('.py'):
            if pathname not in pkgpath:
                pkgpath.append(pathname)
            pathname = pth.join(pathname, '__init__.py')
            if pth.isfile(pathname):
                module.__file__ = pathname

        if pkgpath:
            module.__path__ = pkgpath

        if hasattr(module, '__path__'):
            module.__package__ = str(self)
        else:
            module.__package__ = str(self._parent)

        if module.__file__ and not self._code:
            with open(module.__file__, 'U') as fd:
                self._code = compile(fd.read(), module.__file__, 'exec')
        elif not module.__file__:
            module.__file__ = '(dynamic-module)'

        if self._code not in (None, False, True):
            if '__doc__' in self._code.co_names[0:1]:
                docstr = self._code.co_consts[0].strip()
        module.__doc__ = docstr

        module.__loader__ = self
        self.root._cache[str(self)] = self

    def find_module(self, name=None, path=None):
        if name:
            return self.root._cache.get(name)
        return self

    def load_module(self, name=None):
        loader = self.find_module(name)
        if not loader:
            raise ImportError

        imp.acquire_lock()
        try:
            module = sys.modules[str(loader)] = loader._module
            try:
                if loader._code not in (None, True, False):
                    exec loader._code in module.__dict__
            except:
                sys.modules.pop(module.__name__)
                raise
        finally:
            imp.release_lock()
        return module

    def walk(self, skips=tuple(), rskips=tuple()):
        if self not in skips:
            yield self
        for key, gen1 in sorted(vars(self).iteritems()):
            if not (gen1 in rskips or key.startswith('_')):
                if isinstance(gen1, Namespace):
                    for gen2 in gen1.walk():
                        yield gen2

    def __contains__(self, name):
        return name in self._keys

    def __setitem__(self, key, attr):
        return self._keys.__setitem__(key, attr)

    def __getitem__(self, key):
        return self._keys.__getitem__(key)

    def __delitem__(self, key):
        return self._keys.__delitem__(key)

    def __iter__(self):
        return iter(self._keys)

    @property
    def ctx(self):
        return getattr(self, '_parent', None)

    @property
    def root(self):
        return getattr(self._parent, 'root', self)

    @property
    def nodes(self):
        for node in getattr(self._parent, 'nodes', tuple()):
            yield node
        yield self

    @property
    def addr(self):
        for ns in getattr(self._parent, 'addr', tuple()):
            yield ns
        yield self._ns

    @property
    def pathspec(self):
        for path in getattr(self._parent, 'pathspec', tuple()):
            yield path
        yield self._path or self._ns

    @property
    def path(self):
        return pth.join(*self.pathspec)

    def __call__(self, ns, hint=None, **kwds):
        key = ns.upper()
        if key in self.__dict__:
            return getattr(self, key)
        setattr(self, key, self.__class__(ns, hint, self, **kwds))
        return getattr(self, key)

    def __repr__(self):
        return '<%s.%s: %s>' % (
                __name__,
                self.__class__.__name__,
                ' '.join([
                    ('%s=%r' % x) for x in [
                        ('addr', str(self)),
                        ]]))

    def __str__(self):
        return '.'.join(self.addr)


class Xacto(object):
    """top-level entry point and root Namespace(...)
    """

    #TODO: ./once --plural=arg levels --isa=thing
    # we can point xacto at itself! eg. ./xacto --debug user [...]
    def __call__(self, debug=False, mode='development', loglevel='info'):
        """"""

    config = {
        'store_bools': False,
        }

    trans = {
        # the default __doc__
        #None: '',
        'stuff': None,
        'stupid': None,
        'things?': None,
        'hacks?': None,
        'crap': None,
        'junk': None,
        }

    def __init__(self, ns):
        self._ns = ns
        self._parser = None
        self._subparser = None

        nsroot = self._ns.root
        nspath = self._ns.path
        addr = tuple(self._ns.addr)

        for (path, dirs, files) in os.walk(nspath):
            dirs[:] = sorted(set(dirs) - set(
                d for d in dirs if d.startswith('_')
                ))
            rpath = pth.relpath(path, nspath)
            for name in sorted(files, key=lambda x: pth.splitext(x)[0]):
                if (not name.endswith('.py') or (
                    name.startswith('_') and name != '__init__.py')):
                    continue
                lname = (rpath if name == '__init__.py'
                         else pth.join(rpath, pth.splitext(name)[0]))
                lname = pth.join(*(addr+(lname,)))
                lname = re.sub(
                        '[^0-9A-Za-z/_.]', '_',
                        pth.normpath(lname),
                        ).strip('.').replace('/', '.')
                lname = re.sub('(?<=[.])[0-9]', '_\g<0>', lname)

                node = None
                dyn = list()
                pfx, _, ns = lname.rpartition('.')
                while pfx:
                    node = nsroot._cache.get(pfx)
                    if node:
                        break
                    pfx, _, sfx = pfx.rpartition('.')
                    dyn.append(sfx)

                assert node is not None
                for missing in dyn[::-1]:
                    node = node(missing)

                node = node(ns, pth.join(path, name))

    def find_module(self, name, path=None):
        loader = self._ns.find_module(name)
        if not loader:
            msg = 'invalid %s tool: %s' % (self._ns._ns, name)
            warnings.warn(msg, ImportWarning, stacklevel=2)
        return loader

    def load_module(self, name):
        return self._ns.load_module(name)

    def reify(self):
        orphans = set()
        trans = lambda x: self.trans.get(*([x and str(x).strip()]*2)) or ''
        name_get = lambda x: name_map.get(x.group(0))
        name_map = {
            __name__.lower(): self._ns._ns.lower(),
            __name__.title(): self._ns._ns.title(),
            __name__.upper(): self._ns._ns.upper(),
            }
        name_re = re.compile('|'.join(name_map), re.IGNORECASE)
        kill_re = re.compile('|'.join(
            kv[0] for kv in self.trans.iteritems() if kv[0] and kv[1] is None
            ), re.IGNORECASE)

        #FIXME: impl ctx manager for sys.meta_path/sys.modules
        sys.meta_path[0:0] = [self]
        sys.modules.setdefault(str(self._ns), self._ns._module)
        for ns in self._ns.walk():
            module = __import__(str(ns), fromlist=['*'])
            potentials = getattr(module, '__all__', None)
            gconfig = self.config.copy()
            gconfig.update(getattr(module, '__xacto__', tuple()))
            if potentials is None:
                potentials = tuple(k for k in module.__dict__ if k[0] != '_')
            for key in (None,) + tuple(potentials):
                potential = None
                potential = (
                    module if key is None
                    else getattr(module, key, None)
                    )
                if key and potential and not hasattr(potential, '__call__'):
                    potential = None
                if not potential:
                    continue

                name = (key or ns._ns).lower()
                app = re.sub('[^0-9a-z]+', '-', name).strip('-')
                if not app:
                    continue

                lconfig = gconfig.copy()
                lconfig.update(getattr(potential, '__xacto__', tuple()))

                ctx = (key and ns) or ns.ctx or self
                par = ctx._parser
                sub = ctx._subparser or self
                if par and sub is self:
                    orphans.discard(ctx)
                    sub = ctx._subparser = par.add_subparsers(
                        title='additional modules',
                        metavar='OBJECT',
                        help=':= { .%s }' % ctx,
                        )

                _rawdoc = getattr(potential, '__doc__', None)
                doc = name_re.sub(name_get, str(trans(_rawdoc)))
                doc = list(doc.partition('\n')[::2])
                doc[0] = kill_re.sub('', doc[0])
                doc[0] = re.sub(' {2,}', ' ', doc[0])
                doc = map(trans, doc)
                doc0 = _rawdoc and doc[0] or None
                doc01 = doc[0] or doc[1]
                doc10 = doc[1] or doc[0]

                par = ns._parser = sub.add_parser(app, help=doc01)
                par.description = doc10
                par.epilog = doc0
                par.xacto = self
                par._actions[0].help = argparse.SUPPRESS
                if '-' in app:
                    sub._name_parser_map[app.replace('-', '_')] = par
                if not key:
                    orphans.add(ns)
                    continue

                key = None
                spec = None
                offset = None
                attrs = [(None, 0)]
                if isinstance(potential, type):
                    attrs.append(('__init__', 1))
                    attrs.append(('__new__', 1))
                attrs.append(('__call__', 0))
                for key, offset in attrs:
                    try:
                        attr = key and getattr(potential, key)
                        spec = inspect.getargspec(attr or potential)
                    except OSError:
                        continue
                    except TypeError:
                        break
                    else:
                        break
                if not spec:
                    continue

                nil = object()
                if spec.defaults is None:
                    spec = spec._replace(defaults=[])
                defaults = [nil]*(len(spec.args)-len(spec.defaults))
                defaults.extend(spec.defaults)
                for arg, default in zip(spec.args, defaults):
                    kwds = {
                        'help': trans(None),
                        'dest': arg,
                        'type': str,
                        'metavar': 'STR',
                        'required': True,
                        }
                    if default is not nil:
                        kwds['default'] = default
                        kwds['required'] = False
                        if lconfig.get('store_bools') and isinstance(default, bool):
                            kwds.pop('type')
                            kwds.pop('metavar')
                            kwds['action'] = 'store_true'
                            if default:
                                kwds['action'] = 'store_false'
                        elif default is not None:
                            kwds['type'] = default.__class__
                            kwds['metavar'] = kwds['type'].__name__.upper()
                    if hasattr(default, '_argspec'):
                        arginst = default(default, par, potential, arg, kwds)
                    else:
                        pfx = '-' * min(len(arg), 2)
                        arginst = par.add_argument(pfx + arg, **kwds)
                    setattr(potential, arg, arginst)
                if spec.varargs:
                    arginst = par.add_argument(
                        'xacto_vary',
                        metavar=spec.varargs,
                        nargs='*',
                        )
                par.set_defaults(
                    xacto_ns=ns,
                    xacto_call=potential,
                    xacto_args=spec.args,
                    xacto_vary=list(),
                    )
        #FIXME: this only works because we [currently] preload all modules
        sys.meta_path.remove(self)

        # hide parsers lacking concrete exports
        for ns in orphans:
            if ns.ctx and ns is not ns.ctx:
                ns.ctx._subparser._choices_actions = [
                    c for c in ns.ctx._subparser._choices_actions
                    if c.dest != ns._ns
                    ]

    def add_parser(self, *args, **kwds):
        kwds.pop('help', None)
        kwds['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        self._parser = XactoParser(*args, **kwds)
        return self._parser

    @classmethod
    def from_module(cls, module=None, name=None, path=None, rel=None):
        from glob import glob

        if module is None:
            module = __name__
        if isinstance(module, basestring):
            module = __import__(module, fromlist='*')

        if None in (path, name):
            file = (
                getattr(rel, '__file__', '') or
                getattr(module, '__file__', '')
                )
            if module.__name__ == '__main__':
                file = file or sys.argv[0]
            if file.endswith(('/__init__.py', '/__main__.py')):
                file = file[:-12]
            if file.endswith(('/__init__.pyc', '/__main__.pyc')):
                file = file[:-13]
            if file.endswith(('/__init__.pyo', '/__main__.pyo')):
                file = file[:-13]
            if not file:
                file = 'xacto'
            file = pth.abspath(file)
            file_noext = pth.splitext(file)[0]

        if not name:
            name = pth.basename(file_noext)

        if not path:
            head = pth.dirname(file_noext)
            vfile = pth.join(head, name)
            expands = (
                '%s*tools' % vfile,
                '%s/*tools' % vfile,
                '%s/*tools' % head,
                '%s*tools' % name,
                '*tools',
                )
            for x in expands:
                path = glob(x) or None
                if path:
                    #FIXME: handle multiple sources
                    path = pth.abspath(path[0])
                    break

            if not path:
                import pkg_resources
                for entry_point in pkg_resources.iter_entry_points(
                    'xacto.tools', name,
                    ):
                    # return first entry
                    return cls.from_module(
                        module=module, name=name, rel=entry_point.load(),
                        )

            assert path, 'error: missing toolbox: %r' % (expands,)

        warnings.filterwarnings(
            'default', '', ImportWarning, '^%s([.]|$)' % name,
            )
        return cls(Namespace(name, module, path))

    def main(self, *args, **kwds):
        if not args and not kwds:
            #TODO: use argv[0] to some effect?
            args = sys.argv[1:]

        #from IPython import embed; embed()
        if not self._parser:
            self.reify()

        args_ino, args_idk = self._parser.parse_known_args(args)
        magic = self._parser.prefix_chars
        args_idk = iter(args_idk)
        for argv in args_idk:
            key = value = None
            arg = argv.lstrip(magic) or None
            if arg and arg != argv:
                offset = (len(argv) - len(arg))
                if offset == 1:
                    key, value = arg[0], arg[1:] or True
                if offset == 2:
                    key, value = arg.partition('=')[::2]
                if not key and value:
                    key, value = value, key
                if key and not value:
                    value = True
                if key and value:
                    kwds.setdefault(key, value)
                    continue

            # except via **kwds, '' is an impossible(?) keyword, thereby
            # guaranteed conflict_free(when=keywords, defined=normally).
            #TODO: use '*' instead of '', and avoid:
            #   TypeError: create() got an unexpected keyword argument '*'
            but_wait_theres_more = kwds.setdefault('', list())
            if arg is not None:
                but_wait_theres_more.append(arg)
            but_wait_theres_more.extend(args_idk)
            #...iterator exhausted; loop will end naturally

        kwds.update(args_ino.__dict__)
        ns = kwds.pop('xacto_ns', None)
        if not ns:
            return self._parser.error('no exports found')

        imports = getattr(ns._module, '_imports', None)
        if imports:
            ns._module.__dict__.update(imports())
        call = kwds.pop('xacto_call')
        args = list(
            kwds.pop(x)
            for x in kwds.pop('xacto_args')
            ) + kwds.pop('xacto_vary')
        return call(*args, **kwds)


class XactoParser(argparse.ArgumentParser):

    colors = {
        'USE' : os.isatty(sys.stderr.fileno()) if hasattr(sys.stderr, 'fileno') else False,
        'BOLD'  :'\x1b[01;1m',
        'RED'   :'\x1b[31m',
        'GREEN' :'\x1b[32m',
        'YELLOW':'\x1b[33m',
        'PINK'  :'\x1b[35m',
        'BLUE'  :'\x1b[34m',
        'CYAN'  :'\x1b[36m',
        'NORMAL':'\x1b[0m',
        'cursor_on'  :'\x1b[?25h',
        'cursor_off' :'\x1b[?25l',
        }

    def error(self, message):
        on = off = ''
        if self.colors['USE']:
            on = self.colors['BOLD'] + self.colors['RED']
            off = self.colors['NORMAL']
        self.print_help()
        sys.stderr.write('%serror: %s\n%s' % (on, message, off))
        if self.xacto._ns._module == sys.modules.get('__main__'):
            sys.exit(2)
        return 2


def main(*args, **kwds):
    return Xacto.from_module('__main__').main()


# this module should not be exported!
__all__ = []


if __name__ == '__main__':
    main()
