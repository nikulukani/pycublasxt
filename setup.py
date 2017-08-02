from setuptools import find_packages, setup, Extension
import os

h_files = []

pycublasxt = Extension("pycublasxt", sources=["pycublasxt/pycublasxt.cpp"],
                       depends = h_files,
                       library_dirs = [],
                       include_dirs = ["./pycublasxt", os.environ['CUDA_ROOT'] + "/include"],
                       language="c++", libraries=["stdc++", "cublas", "cudart"],
                       extra_compile_args=['-m64'])
setup(
    name = "pycublasxt",
    version="1.0.0",
    description="Python wrapper for cublasXt",
    author="Nikul Ukani",
    author_email="nhu2001@columbia.edu",
    ext_modules = [pycublasxt],
    packages=find_packages(),
    url="",
)
