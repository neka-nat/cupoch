import os
from setuptools import setup

# Force platform specific wheel
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    # https://stackoverflow.com/a/45150383/1255535
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    print('Warning: cannot import "wheel" package to build platform-specific wheel')
    print('Install the "wheel" package to fix this warning')
    bdist_wheel = None

cmdclass = {'bdist_wheel': bdist_wheel} if bdist_wheel is not None else dict()


# Read requirements.txt
with open('requirements.txt', 'r') as f:
    lines = f.readlines()
install_requires = [line.strip() for line in lines if line]

def find_stubs(package):
    stubs = []
    for root, _, files in os.walk(package):
        for file in files:
            path = os.path.join(root, file).replace(package + os.sep, "", 1)
            stubs.append(path)
    return {package: stubs}

setup(
    author='neka-nat',
    author_email='@PROJECT_EMAIL@',
    classifiers=[
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Environment :: Win32 (MS Windows)",
        "Environment :: X11 Applications",
        "Framework :: Robot Framework",
        "Framework :: Robot Framework :: Library",
        "Framework :: Robot Framework :: Tool",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    description=[
        "Cupoch: Robotics with GPU computing"
    ],
    cmdclass=cmdclass,
    install_requires=install_requires,
    include_package_data=True,
    keywords="robotics point-cloud mesh RGB-D collision visualization",
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # Name of the package on PyPI
    name="@PYPI_PACKAGE_NAME@",
    packages=[
        'cupoch',
    ],
    packages_data=find_stubs("cupoch-stubs"),
    url="@PROJECT_HOME@",
    project_urls={
        'Documentation': '@PROJECT_DOCS@',
        'Source code': '@PROJECT_CODE@',
        'Issues': '@PROJECT_ISSUES@',
    },
    version='@PROJECT_VERSION@',
    zip_safe=False,
)