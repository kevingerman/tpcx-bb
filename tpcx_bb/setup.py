# Copyright (c) 2020, NVIDIA CORPORATION.
from setuptools import find_packages, setup
import os

requirements = [
    "dask", "cudf", "dask_cudf", "cupy", "pandas", "requests", "rmm", "pynvml"
]

qnums = [str(i).zfill(2) for i in range(1, 31)]

package_data={'':['*.json','xbb_tools/*.json'],
              "benchmark_runner": ["benchmark_config.yaml"] }

packages=['xbb_tools']

for root, dir, files in os.walk( 'xbb_tools' ):
    packages.append(root.replace(os.path.sep,'.'))


setup(
    name='xbb_tools',
    version='0.2',
    author='RAPIDS',
    packages=["benchmark_runner", "xbb_tools"],
    package_data={"benchmark_runner": ["benchmark_config.yaml"]},
    entry_points={
        "console_scripts": [
            "daskcluster=xbb_tools.daskcluster:cli"
        ]
    },
    include_package_data=True,
    install_requires=requirements
)
