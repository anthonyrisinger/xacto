# coding: utf-8
"""
Xacto CLI

High-level operational tool encapsulating all others!
"""


from __future__ import absolute_import
from __future__ import print_function


import sys, os
import warnings
import operator
import collections
from os import path as pth

import argparse, pkgutil, types, inspect, imp, re
from collections import defaultdict
from itertools import count


__version__ = '0.8.7'


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

        for key in keys - ikeys:
            self[key] = get(key)
        for key in ikeys - keys:
            del self[key]

        return self


record = mapo.matic(features='attr set')
automap = record.matic(features='auto')


#TODO: unify with DeferredModule
class MockModule(object):

    _index = count(1)

    def __init__(self, name, doc=None, importer=None):
        self.__dict__.update({
            '__loader__': self,
            '__path__': list(),
            '__name__': name,
            '__doc__': doc,
            'gets': list(),
            'sets': list(),
            'index': self._index.next(),
            'importer': importer,
            })

    def __getattr__(self, key):
        if key not in self.gets:
            self.gets.append(key)
        return self

    def __setattr__(self, key, attr):
        if attr not in self.sets:
            self.sets.append(attr)
        return None

    def __repr__(self):
        return '<{}.{}[{}]: {}: gets:{} sets:{}>'.format(
            __name__,
            self.__class__.__name__,
            self.index,
            self.__name__,
            self.gets,
            self.sets,
            )

    def load_module(self, name):
        #TODO: verify works in python3
        # normally loaders are responsible for putting loaded modules into
        # sys.modules... if available, python will prefer the module there
        # over the one returned here (allows modules to replace themselves
        # during import). however, we explicitly DO NOT update sys.modules
        # because we want every import to create fresh mocks!
        return self

    @property
    def iter_imports(self):
        for sub in self.sets:
            yield sub

            for sub2 in sub.iter_imports:
                yield sub2


class DeferredModule(object):

    class dict_cls(dict):

        def __init__(self, *args, **kwds):
            super(DeferredModule.dict_cls, self).__init__(*args, **kwds)
            self.importers = set()

        def __missing__(self, key):
            for importer in self.importers:
                resolved = importer._maybe_import(self, key)
                if resolved:
                    # on-demand loading in py2 only works for code executing at
                    # the module level. a function defined therein is properly
                    # bound (fun.__globals__ is self), but DOES NOT trigger
                    # __missing__ during symbol resolution, ultimately failing
                    # with NameError... as such, xacto preloads all imports
                    # once a tool is "selected".
                    msg = 'early import in {0}: {1}'.format(
                        self['__name__'],
                        resolved.__name__,
                        )
                    warnings.warn(msg, ImportWarning, stacklevel=2)
                    return self[key]

            if '__builtins__' in self and key in self['__builtins__']:
                # Python 2.x seems to do this for us, but no harm doing ourselves.
                return self['__builtins__'][key]

            raise KeyError(key)

        def __setitem__(self, key, item):
            # if it's a placeholder, don't actually save it... the first
            # request will then fire __missing__ instead
            if isinstance(item, MockModule):
                importer = item.importer
                importer._defer_import(self, key, item)
                self.importers.add(importer)
                return None

            return super(DeferredModule.dict_cls, self).__setitem__(key, item)


    def __new__(cls, *args, **kwds):
        self = super(DeferredModule, cls).__new__(cls)
        self.__dict__ = self.dict_cls()
        return self

    def __init__(self, name, doc=None):
        super(DeferredModule, self).__init__()
        self.__name__ = name
        self.__doc__ = doc


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
            module = self._module = DeferredModule(str(self))
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
        else:
            pathname_head, pathname_tail = os.path.split(pathname)
            importer = pkgutil.get_importer(pathname_head)
            try:
                code = importer.get_data(pathname)
            except IOError:
                code = None
            if code:
                module.__file__ = pathname
                self._code = compile(code, pathname, 'exec')

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
                    eval(loader._code, module.__dict__)
            except:
                sys.modules.pop(module.__name__)
                raise
        finally:
            imp.release_lock()
        return module

    def walk(self, skips=tuple(), rskips=tuple()):
        if self not in skips:
            yield self
        for key, gen1 in sorted(vars(self).items()):
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


def _walk(top, topdown=True, onerror=None, followlinks=False):
    if os.path.exists(top):
        return os.walk(top)

    # zipimporter?
    importer = pkgutil.get_importer(top)
    if not hasattr(importer, '_files'):
        return tuple()

    def walker(split):
        for key in sorted(split):
            yield split[key]

    def sorter(value):
        # make sure directories sort properly
        key = value.split('/')
        return key

    split = dict()
    files = sorted(importer._files, key=sorter)
    sprefix = importer.prefix.rstrip('/')
    for key in files:
        if key.startswith(importer.prefix):
            head, tail = os.path.split(key)
            if head not in split:
                fullpath = os.path.join(importer.archive, head)
                split[head] = (fullpath, [], [])
                if head != sprefix:
                    head2, tail2 = os.path.split(head)
                    split[head2][1].append(tail2)
            split[head][2].append(tail)

    return walker(split)


class Xacto(object):
    """top-level entry point and root Namespace(...)
    """

    #TODO: ./once --plural=arg levels --isa=thing
    # we can point xacto at itself! eg. ./xacto --debug user [...]
    def __call__(self, debug=False, mode='development', loglevel='info'):
        """"""

    config = {
        'store_bools': True,
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

    def _defer_import(self, target, key, module):
        cache = self._deferred[(target['__name__'], key)]
        cache.append(module)

    def _maybe_import(self, target, key):
        cache = self._deferred[(target['__name__'], key)]
        if not cache:
            return False

        for module in cache:
            self._deferred_ok = False
            try:
                resolved = __import__(module.__name__, fromlist=module.gets)
                for submodule in module.iter_imports:
                    __import__(submodule.__name__, fromlist=submodule.gets)
                if key in module.gets:
                    resolved = getattr(resolved, key)
            finally:
                self._deferred_ok = True

        target[key] = resolved
        return resolved

    def _import_all(self):
        imports = list()
        #TODO: handle better
        for k, modules in self._deferred.items():
            name, key = k
            for module in modules:
                imports.append((
                    module.index,
                    sys.modules[name],
                    key,
                    ))

        imports.sort()
        for index, module, key in imports:
            self._maybe_import(module.__dict__, key)

    def __init__(self, ns):
        self._ns = ns
        self._parser = None
        self._subparser = None
        self._deferred = defaultdict(list)
        self._deferred_ok = True

        nsroot = self._ns.root
        nspath = self._ns.path
        addr = tuple(self._ns.addr)

        for (path, dirs, files) in _walk(nspath):
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
                if lname in nsroot._cache:
                    # node already exists!
                    continue

                node = None
                dyn = list()
                pfx = lname
                sfx = lname
                while pfx:
                    pfx, sfx = pfx.rpartition('.')[::2]
                    node = nsroot._cache.get(pfx or sfx)
                    if node:
                        break

                    dyn.append(sfx)

                for missing in dyn[::-1]:
                    node = node(missing)

                node = node(sfx, pth.join(path, name))

    def find_module(self, name, path=None):
        loader = self._ns.find_module(name)
        if not loader and self._deferred_ok:
            loader = MockModule(name=name, importer=self)
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
            kv[0] for kv in self.trans.items() if kv[0] and kv[1] is None
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

                potential_call = getattr(potential, '__call__', None)
                potential_self = getattr(potential_call, '__self__', None)
                if potential_self is potential:
                    # a free function, neither bound nor unbound
                    potential_call = potential
                    potential_self = False
                potential_args = None
                potential_defs = None

                _rawdoc = (
                    getattr(potential, '__doc__', None) or
                    getattr(potential_call, '__doc__', None)
                    )
                doc = name_re.sub(name_get, str(trans(_rawdoc)))
                doc = list(doc.partition('\n')[::2])
                doc[0] = kill_re.sub('', doc[0])
                doc[0] = re.sub(' {2,}', ' ', doc[0])
                doc = list(map(trans, doc))
                doc0 = _rawdoc and doc[0] or None
                doc01 = doc[0] or doc[1]
                doc10 = doc[1] or doc[0]

                par = ns._parser = sub.add_parser(app, help=doc01)
                par.description = doc10
                par.epilog = doc0
                par.xacto = self
                par._actions[0].help = argparse.SUPPRESS
                if '-' in app and sub is not self:
                    sub._name_parser_map[app.replace('-', '_')] = par
                if not key:
                    orphans.add(ns)
                    continue

                spec = None
                try:
                    spec = inspect.getargspec(potential_call)
                except TypeError:
                    pass
                except OSError:
                    pass
                if not spec:
                    continue

                nil = object()
                if spec.defaults is None:
                    spec = spec._replace(defaults=[])
                potential_args = spec.args[potential_self is not False:]
                potential_defs = spec.defaults[:]
                potential_diff = len(potential_defs) - len(potential_args)
                if potential_diff > 0:
                    potential_defs = potential_defs[potential_diff:]
                defaults = [nil]*(len(potential_args)-len(potential_defs))
                defaults.extend(potential_defs)
                if spec.varargs:
                    potential_args.append(spec.varargs)
                    defaults.append(nil)
                for arg, default in zip(potential_args, defaults):
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

                    #FIXME: factor this out, almost copypasta with above
                    proto_attr = getattr(potential, arg, None)
                    proto_call = getattr(proto_attr, '__call__', None)
                    proto_self = getattr(proto_call, '__self__', None)
                    if proto_self is proto_attr:
                        proto_call = proto_attr
                        #FIXME: this feels too brittle
                        if getattr(proto_attr, 'im_class', None) is potential:
                            proto_self = potential_self
                    proto_spec = None
                    proto_args = None
                    proto_defs = None
                    proto_doc = None

                    if proto_call:
                        proto_spec = inspect.getargspec(proto_call)
                        if proto_spec.defaults is None:
                            proto_spec = proto_spec._replace(defaults=[])
                        proto_args = proto_spec.args[proto_self is not False:]
                        proto_defs = proto_spec.defaults[:]
                        proto_diff = len(proto_defs) - len(proto_args)
                        if proto_diff > 0:
                            proto_defs = proto_defs[proto_diff:]
                        if proto_args and not proto_spec.varargs:
                            kwds['nargs'] = len(proto_args)
                            if kwds['nargs'] == 1:
                                if proto_defs:
                                    kwds['const'] = proto_defs[0]
                                    kwds['nargs'] = '?'
                                else:
                                    # don't produce a list of one item
                                    kwds['nargs'] = None
                        elif not proto_args and proto_spec.varargs:
                            kwds['nargs'] = '*'
                        elif proto_args and proto_spec.varargs:
                            #TODO: this should really be something like {2,3}
                            # http://bugs.python.org/issue11354
                            kwds['nargs'] = '+'
                        if kwds.get('nargs') is not None:
                            proto_type_args = list(proto_args)
                            if proto_spec.varargs:
                                # special acccessor for *args
                                slicer = slice(len(proto_args), None)
                                getter = property(operator.itemgetter(slicer))
                                proto_type_args.append(proto_spec.varargs)
                            proto_type = collections.namedtuple(
                                proto_call.__name__,
                                proto_type_args,
                                )
                            if proto_spec.varargs:
                                # special acccessor for *args
                                setattr(proto_type, proto_spec.varargs, getter)
                            def proto_type_new_closure(
                                proto_spec=proto_spec,
                                proto_args=proto_args,
                                ):
                                def proto_type_new(cls, *args):
                                    diff = len(proto_args) - len(args)
                                    if diff > 0:
                                        #FIXME: do this differently...?
                                        args += (None,) * diff
                                    return tuple.__new__(cls, args)
                                return proto_type_new
                            def proto_type_repr_closure(
                                name=name,
                                ):
                                def proto_type_repr(self, *args):
                                    qualname = '.'.join([
                                        self.__class__.__module__, name,
                                        self.__class__.__name__, 'type',
                                        ])
                                    fields = ', '.join(
                                        '{0}={1!r}'.format(f, getattr(self, f))
                                        for f in self._fields
                                        )
                                    return '{0}({1})'.format(qualname, fields)
                                return proto_type_repr
                            proto_type.__module__ = str(ns)
                            # originals fail with variable-length input
                            # and repr provides more accurate information
                            proto_type.__new__ = staticmethod(proto_type_new_closure())
                            proto_type.__repr__ = proto_type_repr_closure()
                            #FIXME: this REALLY needs to be object based!
                            # make the generated type available
                            proto_fun = getattr(
                                proto_call, '__func__', proto_call,
                                )
                            setattr(proto_fun, 'type', proto_type)
                        proto_doc = trans(proto_call.__doc__)
                        if proto_doc:
                            kwds['help'] = proto_doc
                        if proto_spec.varargs:
                            #TODO: add support for multiple metavars
                            kwds['metavar'] = proto_spec.varargs.upper()
                        elif len(proto_args) == 1:
                            #TODO: add support for multiple metavars
                            kwds['metavar'] = proto_args[0].upper()
                        #elif proto_args:
                        #    #TODO: add support for multiple metavars
                        #    kwds['metavar'] = proto_args[0].upper()
                        #FIXME: detect/handle better... incompatible signature
                        # eg. cls.__call__(arg=True, ...) and cls.arg(*list)
                        kwds.pop('action', None)

                    pfx = '-' * min(len(arg), 2)
                    if arg == spec.varargs:
                        #TODO: *args (positionals) will ALWAYS be a tuple to
                        # the final __call__(...); maybe we should chunk them
                        # into appropriate groups and call proto_type(*group)?
                        # else *args will ALWAYS be a tuple of length 1 with a
                        # single proto_type(...) object?
                        kwds['nargs'] = kwds.pop('nargs', None) or '*'
                        kwds.pop('required', None)
                        kwds.pop('metavar', None)
                        kwds.pop('dest', None)
                        kwds.pop('type', None)
                        pfx = ''
                    arginst = par.add_argument(pfx + arg, **kwds)

                # False : function
                #  None : method (unbound)
                #     * : method (bound)
                if potential_self is None:
                    potential = potential()

                par.set_defaults(
                    xacto_ns=ns,
                    xacto_call=potential,
                    xacto_args=potential_args,
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
        # TODO: should probably depend on six
        if isinstance(module, (type(u''), type(''))):
            module = __import__(module, fromlist='*')

        # FIXME: fixup __file__??? this initializer is too gnarly :(
        if not hasattr(module, '__file__') and hasattr(rel, '__file__'):
            module.__file__ = rel.__file__

        if name is None:
            try:
                # set by zippy; records the resolved export key/entry
                name = sys.export.key
            except AttributeError:
                pass

        if None in (path, name):
            file = (
                getattr(rel, '__file__', '') or
                getattr(module, '__file__', '')
                )
            if module.__name__ == '__main__':
                file = file or sys.argv[0]
            if file == '-c':
                file = os.path.basename(sys.executable)
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

        if rel is None:
            try:
                rel = __import__(name + '.tools', fromlist='*')
            except ImportError:
                pass
            else:
                # found {name}.tools
                return cls.from_module(
                    module=module, name=rel.__name__, rel=rel,
                    )

        if not path and rel is not None:
            path = file_noext

        if not path:
            head = pth.dirname(file_noext)
            vfile = pth.join(head, name)
            expands = (
                '%s*tools' % vfile,
                '%s/*tools' % vfile,
                '%s/*tools' % head,
                '%s*tools' % name,
                '%s/*tools' % name,
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

        # python2
        self._import_all()

        args = list()
        xacto_call = kwds.pop('xacto_call')
        xacto_args = kwds.pop('xacto_args')
        for arg_name in xacto_args:
            arg = kwds.pop(arg_name)
            arg_new = None
            arg_prep = getattr(xacto_call, arg_name, None)
            arg_type = getattr(arg_prep, 'type', None)
            if not hasattr(arg_prep, '__call__'):
                arg_prep = None
            if not hasattr(arg_type, '__call__'):
                arg_type = None
            if arg_type:
                #TODO: only list objects get unpacked? eliminate this?
                # call this twice because arg_type might fill in any gaps
                if arg is None:
                    arg = tuple()
                else:
                    try:
                        arg = tuple(arg)
                    except TypeError:
                        arg = (arg,)
                arg = arg_type(*arg)
            if arg_prep:
                #TODO: only list objects get unpacked? eliminate this?
                arg_new = arg_prep(*arg) if arg_type else arg_prep(arg)
            if arg_new is not None:
                arg = arg_new
                if arg_type:
                    arg = arg_type(*arg)
            args.append(arg)

        rv = xacto_call(*args, **kwds)
        return rv


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
