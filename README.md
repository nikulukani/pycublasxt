# pycublasxt
This package provides a python interface for the [NVIDIA CublasXt API](http://docs.nvidia.com/cuda/cublas/index.html#using-the-cublasXt-api)

# Installation
1) Make sure the latest version of the CUDA TOOLKIT is installed
2)
```bash
git clone https://github.com/nikulukani/pycublasxt.git
cd pycublasxt
python setup.py install
```

# Usage
An instance of pycublasxt.CublasXt needs to be created and be used to specify the devices first.
```python
from pycublasxt import CublasXt
cublasxt = CublasXt()

cublasxt.CublasXt()

devices = [0,1,2,3]
ngpu = len(devices)
cublasxt.cublasXtDeviceSelect(ngpu,
                              devices)
```

The cublasxt object above can now be used to access all the functions in the [CublasXt Math API](http://docs.nvidia.com/cuda/cublas/index.html#unique_1440937429)
Additionally, the cublasXtSetBlockDim is also exposed. For best performance, this function should be called with an appropriate block size before invoking any
CublasXt Math functions.

You do not have to provide the first argument (handle) for all the exposed functions. The package takes care of creating and destroying the handle and
providing the same to all the methods.

The following [CublasXt datatype constants](http://docs.nvidia.com/cuda/cublas/index.html#cublas-datatypes-reference) can also be accessed through the instantiated object.

```python
cublasxt._CUBLAS_OP['N']
cublasxt._CUBLAS_OP['n']
cublasxt._CUBLAS_OP['T']
cublasxt._CUBLAS_OP['t']
cublasxt._CUBLAS_OP['C']
cublasxt._CUBLAS_OP['c']

cublasxt._CUBLAS_FILL_MODE['L']
cublasxt._CUBLAS_FILL_MODE['l']
cublasxt._CUBLAS_FILL_MODE['U']
cublasxt._CUBLAS_FILL_MODE['u']

cublasxt._CUBLAS_SIDE_MODE['L']
cublasxt._CUBLAS_SIDE_MODE['l']
cublasxt._CUBLAS_SIDE_MODE['R']
cublasxt._CUBLAS_SIDE_MODE['r']

cublasxt._CUBLAS_DIAG['U']
cublasxt._CUBLAS_DIAG['u']
cublasxt._CUBLAS_DIAG['N']
cublasxt._CUBLAS_DIAG['n']
```

# Example

## Comparing cublasXt<t>symm with np.dot
```python
from __future__ import print_function
import numpy as np
import time
from pycublasxt import CublasXt

M = 9000
N = 12000
devices = [0,1,2,3]     
ngpu = len(devices)
nb = 3000

cublasxt = CublasXt()

cublasxt.cublasXtDeviceSelect(ngpu,
                              devices)

cublasxt.cublasXtSetBlockDim(nb)

dtype_func_map = {
    np.double: cublasxt.cublasXtDsymm,
    np.float64:  cublasxt.cublasXtDsymm,
    np.float32:  cublasxt.cublasXtSsymm,
    np.complex128:  cublasxt.cublasXtZsymm,
    np.complex64:  cublasxt.cublasXtCsymm
}

for dtype in [np.float32, np.double, np.complex64, np.complex128]:
    func = dtype_func_map[dtype]

    if dtype in [np.complex64, np.complex128]:
        real_dtype = np.float32 if dtype==np.complex64 else np.float64
        a = (np.random.rand(M,1).astype(real_dtype) + \
              1j*np.random.rand(M,1).astype(real_dtype))
        a = np.dot(a, a.T).astype(dtype, order='F')
        b = (np.random.rand(M,N).astype(real_dtype) + \
              1j*np.random.rand(M,N).astype(real_dtype)).astype(dtype, order='F')
        c = np.empty((M,N), dtype=dtype, order='F')
    else:
        a = np.random.rand(M,1).astype(dtype, order='F')
        a = np.dot(a, a.T).astype(dtype, order='F')
        b = np.random.rand(M,N).astype(dtype, order='F')
        c = np.empty((M,N), dtype=dtype, order='F')
    

    t = time.time()
    func(cublasxt._CUBLAS_SIDE_MODE['L'],
         cublasxt._CUBLAS_FILL_MODE['U'],
         M, N,
         dtype(1), a.ctypes.data, M, b.ctypes.data, M,
         dtype(0), c.ctypes.data, M)

    print(dtype, " Computation time cublasxt: ", time.time() - t)

    t = time.time()
    cnp = np.dot(a, b)
    print (dtype, " Computation time numpy: ", time.time() - t)

    print ("All close - ", np.allclose(cnp, c))
```

Output
```
<type 'numpy.float32'>  Computation time cublasxt:  0.318243026733
<type 'numpy.float32'>  Computation time numpy:  2.52179503441
All close -  True
<type 'numpy.float64'>  Computation time cublasxt:  0.645288944244
<type 'numpy.float64'>  Computation time numpy:  5.39096403122
All close -  True
<type 'numpy.complex64'>  Computation time cublasxt:  0.569852828979
<type 'numpy.complex64'>  Computation time numpy:  9.36415100098
All close -  True
<type 'numpy.complex128'>  Computation time cublasxt:  1.15235900879
<type 'numpy.complex128'>  Computation time numpy:  19.7785379887
All close -  True
```
