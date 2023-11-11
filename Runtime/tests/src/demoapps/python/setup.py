'''

@author: Steven Tang <steven.tang@bytedance.com>
'''
from skbuild import setup

#https://docs.python.org/3/distutils/apiref.html
setup(
    name="testextensionmodule",
    version="1.2.5",
    packages=["testextensionmodule"],
    package_dir={"": ""},
    cmake_install_dir="testextensionmodule",
)
