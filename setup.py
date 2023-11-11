'''
@author: Steven Tang <steven.tang@bytedance.com>
'''
from skbuild import setup

setup(
    name="MLInsight",
    version="1.0.0",
    packages=["mlinsight"],
    package_dir={"mlinsight": "Runtime/src/trace/hook/python"},
    cmake_install_dir="Runtime/src/trace/hook/python",
    cmake_args=['-DCMAKE_BUILD_TYPE=Debug']
)
