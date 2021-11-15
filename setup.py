from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if '.png' not in x]
long_description = ''.join(lines)

setup(
    name="rb_bench",
    packages=[
        package for package in find_packages() if package.startswith("rb_bench")
    ],
    install_requires=[
        "robosuite==1.3.1",
        "gtimer",
        "gym==0.21.0",
        "gtimer==1.0.0b5",
        "h5py==3.5.0",
        "python-dateutil==2.8.2",
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3.8',
    description="rb_bench: Minimal repo for running SAC benchmark experiments on robosuite",
    author="Josiah Wong",
    url="https://github.com/ARISE-Initiative/robosuite",
    author_email="yukez@cs.utexas.edu",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type='text/markdown'
)