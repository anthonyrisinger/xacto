from setuptools import setup
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

# Read the version number from a source file.
# Why read it, and not import?
# see https://groups.google.com/d/topic/pypa-dev/0PkjVpcxTzQ/discussion
def find_version(*file_paths):
    # Open in Latin-1 so that we avoid encoding errors.
    # Use codecs.open for Python 2 compatibility
    with codecs.open(os.path.join(here, *file_paths), 'r', 'latin1') as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]',
                              version_file, re.M)
    if version_match:
        return version_match.group(1)

    raise RuntimeError('Unable to find version string.')


version = find_version('xacto', '__init__.py')

# Get the long description from the relevant file
with codecs.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    zip_safe=False,#XXX
    name='xacto',
    version=version,
    description='Generate command-line interfaces (CLI) by introspecting callables',
    long_description=long_description,

    # The project URL.
    url='http://github.com/xtfxme/xacto',
    download_url='https://github.com/xtfxme/xacto/archive/{0}.zip'.format(version),

    # Author details
    author='C Anthony Risinger',
    author_email='c@anthonyrisinger.com',

    # Choose your license
    license='BSD',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Environment :: Console',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Pick your license as you wish (should match 'license' above)
        'License :: OSI Approved :: BSD License',

        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        ],

    # What does your project relate to?
    keywords='cli commandline command introspection generate',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages.
    test_suite='xacto.tests',
    packages=[
        'xacto',
        ],

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed.
    install_requires = [
        #'mapo'
        ],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'xacto': [
            ],
        },

    # To provide executable scripts, use entry points in preference to the
    # 'scripts' keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'xacto=xacto:main',
            ],
        },
    )
