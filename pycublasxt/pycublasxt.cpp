#include <Python.h>
#include <structmember.h>
#include <list>

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <cuComplex.h>

#include <cassert>

// TODO: Error messages from status
// TODO: Documentation

using namespace std;

// Custom converters for converting PyComplex or np.complex64/128
// to cuComplex/cuDoubleComplex;
int cuDoubleComplexConverter(PyObject* pyobj, cuDoubleComplex *val){
  double r,i;
  PyObject* robj;
  PyObject* iobj;
  robj = PyObject_GetAttrString(pyobj, "real");
  iobj = PyObject_GetAttrString(pyobj, "imag");
  r = PyFloat_AsDouble(robj);
  i = PyFloat_AsDouble(iobj);
  *val = make_cuDoubleComplex(r, i);
  Py_XDECREF(robj);
  Py_XDECREF(iobj);
  return 1;
}

int cuComplexConverter(PyObject* pyobj, cuComplex *val){
  float r,i;
  PyObject* robj;
  PyObject* iobj;
  robj = PyObject_GetAttrString(pyobj, "real");
  iobj = PyObject_GetAttrString(pyobj, "imag");
  r = (float) PyFloat_AsDouble(robj);
  i = (float) PyFloat_AsDouble(iobj);
  *val = make_cuComplex(r, i);
  Py_XDECREF(robj);
  Py_XDECREF(iobj);
  return 1;
 }

const char *CUBLAS_ERRORS[] = {
  "",
  "Error: CUBLASXT NOT INITIALIZED",
  "",
  "Error: CUBLASXT ALLOCATION FAILED",
  "",
  "",
  "",
  "Error: INVALID VALUE",
  "Error: DEVICE ARCHITECTURE NOT SUPPORTED",
  "",
  "",
  "Error: CUBLASXT MAPPING ERROR",
  "Error: CUBLASXT EXECUTION FAILED",
  "Error: CUBLASXT INTERNAL ERROR",
  "Error: CUBLAS_STATUS_NOT_SUPPORTED",
  "Error: CUBLASXT LICENSE ERROR",
};

typedef struct {
  PyObject_HEAD
  //list<void *> *alloc_mems;
  cublasStatus_t status;
  cublasXtHandle_t handle;
  PyObject* _CUBLAS_OP;
  PyObject* _CUBLAS_FILL_MODE;
  PyObject* _CUBLAS_DIAG;
  PyObject* _CUBLAS_SIDE_MODE;
  int validHandle;
} cublasXt;

/*
static void free_all(cublasXt* self){
  list<void *>::iterator it = self->alloc_mems->begin();
  while ( it != self->alloc_mems->end() ) {
    free(*it);
    it = self->alloc_mems->erase(it);
  }
}
*/
static void finalize(cublasXt* self){
  if(self->validHandle){
    cublasXtDestroy(self->handle);
    self->validHandle = 0;
  }
}

static void
cublasXt_dealloc(cublasXt* self)
{
  Py_XDECREF(self->_CUBLAS_OP);
  Py_XDECREF(self->_CUBLAS_FILL_MODE);
  Py_XDECREF(self->_CUBLAS_DIAG);
  Py_XDECREF(self->_CUBLAS_SIDE_MODE);
  //free_all(self);
  finalize(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
static PyObject *
cublasXt_free(cublasXt* self, PyObject *args)
{
  int ret = -1;
  PyObject *obj;
  if(!PyArg_ParseTuple(args, "O", &obj))
    return NULL;
  void *ptr = PyLong_AsVoidPtr(obj);
  list<void *>::iterator it = self->alloc_mems->begin();
  while ( it != self->alloc_mems->end() ) {
    if(*it==ptr){
      PyMem_Free(*it);
      it = self->alloc_mems->erase(it);
      ret = 0;
      break;
    }
    it++;
  }
  return PyLong_FromLong(ret);

}

static PyObject *
cublasXt_free_all(cublasXt* self, PyObject *args)
{
  free_all(self);
  return PyLong_FromLong(0);
}
*/
static PyObject *
cublasXt_finalize(cublasXt* self, PyObject *args)
{
  finalize(self);
  return PyLong_FromLong(0);
}


static PyObject *
cublasXt_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  cublasXt *self;
  PyObject* zero = PyLong_FromLong(0);
  PyObject* one = PyLong_FromLong(1);
  PyObject* two = PyLong_FromLong(2);
  
  self = (cublasXt *)type->tp_alloc(type, 0);
  self->_CUBLAS_OP = PyDict_New();
  self->_CUBLAS_FILL_MODE = PyDict_New();
  self->_CUBLAS_DIAG = PyDict_New();
  self->_CUBLAS_SIDE_MODE = PyDict_New();

  PyDict_SetItemString(self->_CUBLAS_OP, "n", zero);
  PyDict_SetItemString(self->_CUBLAS_OP, "N", zero);;
  PyDict_SetItemString(self->_CUBLAS_OP, "t", one);
  PyDict_SetItemString(self->_CUBLAS_OP, "T", one);
  PyDict_SetItemString(self->_CUBLAS_OP, "c", two);
  PyDict_SetItemString(self->_CUBLAS_OP, "C", two);

  PyDict_SetItemString(self->_CUBLAS_FILL_MODE, "l", zero);
  PyDict_SetItemString(self->_CUBLAS_FILL_MODE, "L", zero);;
  PyDict_SetItemString(self->_CUBLAS_FILL_MODE, "u", one);
  PyDict_SetItemString(self->_CUBLAS_FILL_MODE, "U", one);

  PyDict_SetItemString(self->_CUBLAS_DIAG, "n", zero);
  PyDict_SetItemString(self->_CUBLAS_DIAG, "N", zero);;
  PyDict_SetItemString(self->_CUBLAS_DIAG, "u", one);
  PyDict_SetItemString(self->_CUBLAS_DIAG, "U", one);

  PyDict_SetItemString(self->_CUBLAS_SIDE_MODE, "l", zero);
  PyDict_SetItemString(self->_CUBLAS_SIDE_MODE, "L", zero);;
  PyDict_SetItemString(self->_CUBLAS_SIDE_MODE, "r", one);
  PyDict_SetItemString(self->_CUBLAS_SIDE_MODE, "R", one);
  
  Py_XDECREF(zero);
  Py_XDECREF(one);
  Py_XDECREF(two);
  
  self->status = cublasXtCreate( &(self->handle) );
  self->validHandle = 1;
  //self->alloc_mems = new list<void *>();
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return (PyObject *)self;
}

static PyObject *
cublasXt_deviceSelect(cublasXt* self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
    
  int ngpu;
  PyObject *pyDevices;
  if(!PyArg_ParseTuple(args, "iO", &ngpu, &pyDevices))
    return NULL;
  assert(ngpu == PySequence_Size(pyDevices));

  int *devices = new int[ngpu];
  PyObject* temp;
  for(int i=0 ; i<ngpu ; i++ ){
    temp = PySequence_GetItem(pyDevices, i);
    devices[i] = (int) PyLong_AsLong(temp);
    Py_XDECREF(temp);
  }
  self->status  = cublasXtDeviceSelect(self->handle, ngpu, devices);
  delete [] devices;
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_setBlockDim(cublasXt* self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  long nb;
  if( !PyArg_ParseTuple(args, "l", &nb) )
    return NULL;
  self->status  = cublasXtSetBlockDim(self->handle, nb);
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

/*
static PyObject *
cublasXt_calloc(cublasXt* self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  long long size;
  long long nelems;
  if( !PyArg_ParseTuple(args, "LL", &nelems, &size) )
    return NULL;
  void *mem = calloc(nelems, size);
  if (mem == 0){
    PyErr_SetString(PyExc_RuntimeError, "!!!! Error allocating memory \n");
    return NULL;
  }
  self->alloc_mems->push_back(mem);
  return PyLong_FromVoidPtr( mem );
}

static PyObject *
cublasXt_malloc(cublasXt* self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  long long size;
  if( !PyArg_ParseTuple(args, "L", &size) )
    return NULL;

  void *mem = malloc(size);
  if (mem == 0){
    PyErr_SetString(PyExc_RuntimeError, "!!!! Error allocating memory \n");
    return NULL;
  }
  self->alloc_mems->push_back(mem);
  return PyLong_FromVoidPtr( mem );
}
*/
static PyObject *
cublasXt_Sgemm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int transa, transb;
  long long m, n, k, lda, ldb, ldc;
  float alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLLfOLOLfOL",
                        &transa,         // transa
                        &transb,         // transb
                        &m,              // m
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* B_p = (float *) PyLong_AsVoidPtr(B);
  float* C_p = (float *) PyLong_AsVoidPtr(C);

  self->status = cublasXtSgemm(self->handle,
                               (cublasOperation_t) transa,         // transa
                               (cublasOperation_t) transb,         // transb
                               m,              // m
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dgemm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int transa, transb;
  long long m, n, k, lda, ldb, ldc;
  double alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLLdOLOLdOL",
                       &transa,         // transa
                       &transb,         // transb
                       &m,              // m
                       &n,              // n
                       &k,              // k
                       &alpha,         // *alpha
                       &A,            // *A
                       &lda,            // lda
                       &B,            // *B
                       &ldb,            // ldb
                       &beta,          // *beta
                       &C,            // *C
                       &ldc) )           // ldc
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* B_p = (double *) PyLong_AsVoidPtr(B);
  double* C_p = (double *) PyLong_AsVoidPtr(C);

  self->status = cublasXtDgemm(self->handle,
                               (cublasOperation_t) transa,         // transa
                               (cublasOperation_t) transb,         // transb
                               m,              // m
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Cgemm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int transa, transb;
  long long m, n, k, lda, ldb, ldc;
  cuComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLLO&OLOLO&OL",
                        &transa,         // transa
                        &transb,         // transb
                        &m,              // m
                        &n,              // n
                        &k,              // k
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCgemm(self->handle,
                               (cublasOperation_t) transa,         // transa
                               (cublasOperation_t) transb,         // transb
                               m,              // m
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zgemm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int transa, transb;
  long long m, n, k, lda, ldb, ldc;
  cuDoubleComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLLO&OLOLO&OL",
                        &transa,         // transa
                        &transb,         // transb
                        &m,              // m
                        &n,              // n
                        &k,              // k
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuDoubleComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtZgemm(self->handle,
                               (cublasOperation_t) transa,         // transa
                               (cublasOperation_t) transb,         // transb
                               m,              // m
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}


static PyObject *
cublasXt_Chemm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, lda, ldb, ldc;
  cuComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &side,         // transa
                        &uplo,         // transb
                        &m,              // m
                        &n,              // n
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtChemm(self->handle,
                               (cublasSideMode_t) side,         // transa
                               (cublasFillMode_t) uplo,         // transb
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zhemm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, lda, ldb, ldc;
  cuDoubleComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &side,         // transa
                        &uplo,         // transb
                        &m,              // m
                        &n,              // n
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuDoubleComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtZhemm(self->handle,
                               (cublasSideMode_t) side,         // transa
                               (cublasFillMode_t) uplo,         // transb
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc
  
  if( self->status != CUBLAS_STATUS_SUCCESS){
    printf("ERROR %d\n",self->status);
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ssymm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, lda, ldb, ldc;
  float alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLfOLOLfOL",
                        &side,         // transa
                        &uplo,         // transb
                        &m,              // m
                        &n,              // n
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* B_p = (float *) PyLong_AsVoidPtr(B);
  float* C_p = (float *) PyLong_AsVoidPtr(C);

  self->status = cublasXtSsymm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dsymm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, lda, ldb, ldc;
  double alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLdOLOLdOL",
                       &side,         // side
                       &uplo,         // uplo
                       &m,              // m
                       &n,              // n
                       &alpha,         // *alpha
                       &A,            // *A
                       &lda,            // lda
                       &B,            // *B
                       &ldb,            // ldb
                       &beta,          // *beta
                       &C,            // *C
                       &ldc) )           // ldc
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* B_p = (double *) PyLong_AsVoidPtr(B);
  double* C_p = (double *) PyLong_AsVoidPtr(C);

  self->status = cublasXtDsymm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Csymm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, lda, ldb, ldc;
  cuComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &side,         // side
                        &uplo,         // uplo
                        &m,              // m
                        &n,              // n
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCsymm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zsymm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, lda, ldb, ldc;
  cuDoubleComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &side,         // side
                        &uplo,         // uplo
                        &m,              // m
                        &n,              // n
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuDoubleComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtZsymm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ssyrk(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldc;
  float alpha, beta;
  PyObject* A;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLfOLfOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* C_p = (float *) PyLong_AsVoidPtr(C);

  self->status = cublasXtSsyrk(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dsyrk(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldc;
  double alpha, beta;
  PyObject* A;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLdOLdOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* C_p = (double *) PyLong_AsVoidPtr(C);

  self->status = cublasXtDsyrk(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Csyrk(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo; 
  long long n,k, lda, ldc;
  cuComplex alpha, beta;
  PyObject* A;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLO&OLO&OL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &cuComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCsyrk(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zsyrk(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldc;
  cuDoubleComplex alpha, beta;
  PyObject* A;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLO&OLO&OL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &cuDoubleComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtZsyrk(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, "CUBLAS_ERRORS[self->status]");
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ssyr2k(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  float alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLfOLOLfOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* B_p = (float *) PyLong_AsVoidPtr(B);
  float* C_p = (float *) PyLong_AsVoidPtr(C);

  self->status = cublasXtSsyr2k(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dsyr2k(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  double alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLdOLOLdOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* B_p = (double *) PyLong_AsVoidPtr(B);
  double* C_p = (double *) PyLong_AsVoidPtr(C);

  self->status = cublasXtDsyr2k(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Csyr2k(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k,lda, ldb, ldc;
  cuComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &cuComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCsyr2k(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zsyr2k(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  cuDoubleComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuDoubleComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);
  

  self->status = cublasXtZsyr2k(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ssyrkx(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  float alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLfOLOLfOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* B_p = (float *) PyLong_AsVoidPtr(B);
  float* C_p = (float *) PyLong_AsVoidPtr(C);

  self->status = cublasXtSsyrkx(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dsyrkx(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  double alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLdOLOLdOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* B_p = (double *) PyLong_AsVoidPtr(B);
  double* C_p = (double *) PyLong_AsVoidPtr(C);

  self->status = cublasXtDsyrkx(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Csyrkx(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  cuComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &cuComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCsyrkx(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zsyrkx(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  cuDoubleComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OLOLO&OL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &cuDoubleComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);
  

  self->status = cublasXtZsyrkx(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, "!!!! CUBLASXT Error \n");
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Cherk(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldc;
  float alpha, beta;
  PyObject* A;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLfOLfOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCherk(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zherk(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldc;
  double alpha, beta;
  PyObject* A;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLdOLdOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtZherk(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, "CUBLAS_ERRORS[self->status]");
    return NULL;
  }
  return PyLong_FromLong(self->status);
}


static PyObject *
cublasXt_Cher2k(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  cuComplex alpha;
  float beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OLOLfOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCher2k(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zher2k(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  cuDoubleComplex alpha;
  double beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OLOLdOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);
  

  self->status = cublasXtZher2k(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Cherkx(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  cuComplex alpha;
  float beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OLOLfOL",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,
                        &ldb,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCherkx(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,
                               ldb,
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zherkx(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int trans, uplo;
  long long n,k, lda, ldb, ldc;
  cuDoubleComplex alpha;
  double beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiLLO&OiOidOi",
                        &uplo,         // uplo
                        &trans,         // trans
                        &n,              // n
                        &k,              // k
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,            // ldb
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);
  

  self->status = cublasXtZherkx(self->handle,
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,         // trans
                               n,              // n
                               k,              // k
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, "!!!! CUBLASXT Error \n");
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Strsm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb;
  long long m, n;
  float alpha;
  PyObject* A;
  PyObject* B;

  if (!PyArg_ParseTuple(args, "iiiiLLfOLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb ) )          // ldb
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* B_p = (float *) PyLong_AsVoidPtr(B);
      
  self->status = cublasXtStrsm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb);            // ldb
      
      
      
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dtrsm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb;
  long long m, n;
  double alpha;
  PyObject* A;
  PyObject* B;

  if (!PyArg_ParseTuple(args, "iiiiLLdOLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb ) )          // ldb
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* B_p = (double *) PyLong_AsVoidPtr(B);
      
  self->status = cublasXtDtrsm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb);            // ldb
      
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ctrsm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb;
  long long m, n;
  cuComplex alpha;
  PyObject* A;
  PyObject* B;

  if (!PyArg_ParseTuple(args, "iiiiLLO&OLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb ) )          // ldb
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
      
  self->status = cublasXtCtrsm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb);            // ldb
      
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ztrsm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb;
  long long m, n;
  cuDoubleComplex alpha;
  PyObject* A;
  PyObject* B;

  if (!PyArg_ParseTuple(args, "iiiiLLO&OLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb ) )          // ldb
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
      
  self->status = cublasXtZtrsm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb);            // ldb
      
      
      
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Strmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb, ldc;
  long long m, n;
  float alpha;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiiiLLfOLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,          // ldb
                        &C,
                        &ldc) )
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* B_p = (float *) PyLong_AsVoidPtr(B);
  float* C_p = (float *) PyLong_AsVoidPtr(C);
      
  self->status = cublasXtStrmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               C_p,
                               ldc);
      
      
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dtrmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb, ldc;
  long long m, n;
  double alpha;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if (!PyArg_ParseTuple(args, "iiiiLLdOLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,          // ldb
                        &C,
                        &ldc) )
                        
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* B_p = (double *) PyLong_AsVoidPtr(B);
  double* C_p = (double *) PyLong_AsVoidPtr(C);
      
  self->status = cublasXtDtrmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               C_p,
                               ldc);
    
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ctrmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb, ldc;
  long long m, n;
  cuComplex alpha;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiiiLLO&OLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,          // ldb
                        &C,
                        &ldc))
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);
      
  self->status = cublasXtCtrmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               C_p,
                               ldc);
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Ztrmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo, trans, diag;
  long long lda, ldb, ldc;
  long long m, n;
  cuDoubleComplex alpha;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiiiLLO&OLOL",
                        &side,         // transa
                        &uplo,         // transb
                        &trans,
                        &diag,
                        &m,              // m
                        &n,              // n
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &lda,            // lda
                        &B,            // *B
                        &ldb,          // ldb
                        &C,
                        &ldc))
    return NULL;
  
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);
      
  self->status = cublasXtZtrmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               (cublasOperation_t) trans,
                               (cublasDiagType_t) diag,
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               lda,            // lda
                               B_p,            // *B
                               ldb,            // ldb
                               C_p,
                               ldc);
      
      
  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Sspmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, ldb, ldc;
  float alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;

  if (!PyArg_ParseTuple(args, "iiLLfOOLfOL",
                        &side,         // transa
                        &uplo,         // transb
                        &m,              // m
                        &n,              // n
                        &alpha,         // *alpha
                        &A,            // *A
                        &B,            // *B
                        &ldb,            // ldb
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  float* A_p = (float *) PyLong_AsVoidPtr(A);
  float* B_p = (float *) PyLong_AsVoidPtr(B);
  float* C_p = (float *) PyLong_AsVoidPtr(C);

  self->status = cublasXtSspmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Dspmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, ldb, ldc;
  double alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLdOOLdOL",
                       &side,         // side
                       &uplo,         // uplo
                       &m,              // m
                       &n,              // n
                       &alpha,         // *alpha
                       &A,            // *A
                       &B,            // *B
                       &ldb,            // ldb
                       &beta,          // *beta
                       &C,            // *C
                       &ldc) )           // ldc
    return NULL;
  double* A_p = (double *) PyLong_AsVoidPtr(A);
  double* B_p = (double *) PyLong_AsVoidPtr(B);
  double* C_p = (double *) PyLong_AsVoidPtr(C);

  self->status = cublasXtDspmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Cspmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, ldb, ldc;
  cuComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLO&OOLO&OL",
                        &side,         // side
                        &uplo,         // uplo
                        &m,              // m
                        &n,              // n
                        &cuComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &B,            // *B
                        &ldb,            // ldb
                        &cuComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuComplex* A_p = (cuComplex *) PyLong_AsVoidPtr(A);
  cuComplex* B_p = (cuComplex *) PyLong_AsVoidPtr(B);
  cuComplex* C_p = (cuComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtCspmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyObject *
cublasXt_Zspmm(cublasXt *self, PyObject *args)
{
  if(!self->validHandle){
    PyErr_SetString(PyExc_RuntimeError, "No valid handle. Please instantiate a new object of this class.");
    return NULL;
  }
  int side, uplo;
  long long m, n, ldb, ldc;
  cuDoubleComplex alpha, beta;
  PyObject* A;
  PyObject* B;
  PyObject* C;
  
  if( !PyArg_ParseTuple(args, "iiLLO&OOLO&OL",
                        &side,         // side
                        &uplo,         // uplo
                        &m,              // m
                        &n,              // n
                        &cuDoubleComplexConverter,
                        &alpha,         // *alpha
                        &A,            // *A
                        &B,            // *B
                        &ldb,            // ldb
                        &cuDoubleComplexConverter,
                        &beta,          // *beta
                        &C,            // *C
                        &ldc) )           // ldc
    return NULL;
  cuDoubleComplex* A_p = (cuDoubleComplex *) PyLong_AsVoidPtr(A);
  cuDoubleComplex* B_p = (cuDoubleComplex *) PyLong_AsVoidPtr(B);
  cuDoubleComplex* C_p = (cuDoubleComplex *) PyLong_AsVoidPtr(C);

  self->status = cublasXtZspmm(self->handle,
                               (cublasSideMode_t) side,         // side
                               (cublasFillMode_t) uplo,         // uplo
                               m,              // m
                               n,              // n
                               &alpha,         // *alpha
                               A_p,            // *A
                               B_p,            // *B
                               ldb,            // ldb
                               &beta,          // *beta
                               C_p,            // *C
                               ldc);           // ldc

  if( self->status != CUBLAS_STATUS_SUCCESS){
    PyErr_SetString(PyExc_RuntimeError, CUBLAS_ERRORS[self->status]);
    return NULL;
  }
  return PyLong_FromLong(self->status);
}

static PyMethodDef cublasXt_methods[] = {
  /*{"malloc", (PyCFunction) cublasXt_malloc, METH_VARARGS, ""},
  {"calloc", (PyCFunction) cublasXt_calloc, METH_VARARGS, ""},
  {"free", (PyCFunction) cublasXt_free, METH_VARARGS, ""},*/
  {"cublasXtDeviceSelect", (PyCFunction) cublasXt_deviceSelect, METH_VARARGS, ""},
  {"cublasXtSetBlockDim", (PyCFunction) cublasXt_setBlockDim, METH_VARARGS, ""},
  {"cublasXtSgemm", (PyCFunction) cublasXt_Sgemm, METH_VARARGS, ""},
  {"cublasXtDgemm", (PyCFunction) cublasXt_Dgemm, METH_VARARGS, ""},
  {"cublasXtCgemm", (PyCFunction) cublasXt_Cgemm, METH_VARARGS, ""},
  {"cublasXtZgemm", (PyCFunction) cublasXt_Zgemm, METH_VARARGS, ""},
  {"cublasXtChemm", (PyCFunction) cublasXt_Chemm, METH_VARARGS, ""},
  {"cublasXtZhemm", (PyCFunction) cublasXt_Zhemm, METH_VARARGS, ""},
  {"cublasXtSsymm", (PyCFunction) cublasXt_Ssymm, METH_VARARGS, ""},
  {"cublasXtDsymm", (PyCFunction) cublasXt_Dsymm, METH_VARARGS, ""},
  {"cublasXtCsymm", (PyCFunction) cublasXt_Csymm, METH_VARARGS, ""},
  {"cublasXtZsymm", (PyCFunction) cublasXt_Zsymm, METH_VARARGS, ""},
  {"cublasXtSsyrk", (PyCFunction) cublasXt_Ssyrk, METH_VARARGS, ""},
  {"cublasXtDsyrk", (PyCFunction) cublasXt_Dsyrk, METH_VARARGS, ""},
  {"cublasXtCsyrk", (PyCFunction) cublasXt_Csyrk, METH_VARARGS, ""},
  {"cublasXtZsyrk", (PyCFunction) cublasXt_Zsyrk, METH_VARARGS, ""},
  {"cublasXtSsyr2k", (PyCFunction) cublasXt_Ssyr2k, METH_VARARGS, ""},
  {"cublasXtDsyr2k", (PyCFunction) cublasXt_Dsyr2k, METH_VARARGS, ""},
  {"cublasXtCsyr2k", (PyCFunction) cublasXt_Csyr2k, METH_VARARGS, ""},
  {"cublasXtZsyr2k", (PyCFunction) cublasXt_Zsyr2k, METH_VARARGS, ""},
  {"cublasXtSsyrkx", (PyCFunction) cublasXt_Ssyrkx, METH_VARARGS, ""},
  {"cublasXtDsyrkx", (PyCFunction) cublasXt_Dsyrkx, METH_VARARGS, ""},
  {"cublasXtCsyrkx", (PyCFunction) cublasXt_Csyrkx, METH_VARARGS, ""},
  {"cublasXtZsyrkx", (PyCFunction) cublasXt_Zsyrkx, METH_VARARGS, ""},
  {"cublasXtCherk", (PyCFunction) cublasXt_Cherk, METH_VARARGS, ""},
  {"cublasXtZherk", (PyCFunction) cublasXt_Zherk, METH_VARARGS, ""},
  {"cublasXtCher2k", (PyCFunction) cublasXt_Cher2k, METH_VARARGS, ""},
  {"cublasXtZher2k", (PyCFunction) cublasXt_Zher2k, METH_VARARGS, ""},
  {"cublasXtCherkx", (PyCFunction) cublasXt_Cherkx, METH_VARARGS, ""},
  {"cublasXtZherkx", (PyCFunction) cublasXt_Zherkx, METH_VARARGS, ""},
  {"cublasXtStrsm", (PyCFunction) cublasXt_Strsm, METH_VARARGS, ""},
  {"cublasXtDtrsm", (PyCFunction) cublasXt_Dtrsm, METH_VARARGS, ""},
  {"cublasXtCtrsm", (PyCFunction) cublasXt_Ctrsm, METH_VARARGS, ""},
  {"cublasXtZtrsm", (PyCFunction) cublasXt_Ztrsm, METH_VARARGS, ""},
  {"cublasXtStrmm", (PyCFunction) cublasXt_Strmm, METH_VARARGS, ""},
  {"cublasXtDtrmm", (PyCFunction) cublasXt_Dtrmm, METH_VARARGS, ""},
  {"cublasXtCtrmm", (PyCFunction) cublasXt_Ctrmm, METH_VARARGS, ""},
  {"cublasXtZtrmm", (PyCFunction) cublasXt_Ztrmm, METH_VARARGS, ""},
  {"cublasXtSspmm", (PyCFunction) cublasXt_Sspmm, METH_VARARGS, ""},
  {"cublasXtDspmm", (PyCFunction) cublasXt_Dspmm, METH_VARARGS, ""},
  {"cublasXtCspmm", (PyCFunction) cublasXt_Cspmm, METH_VARARGS, ""},
  {"cublasXtZspmm", (PyCFunction) cublasXt_Zspmm, METH_VARARGS, ""},
  
  /*{"free_all", (PyCFunction) cublasXt_free_all, METH_NOARGS, ""},*/
  {"finalize", (PyCFunction) cublasXt_finalize, METH_NOARGS, ""},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};

static PyMethodDef pycublasxt_methods[] = {
  {NULL, NULL, 0, NULL}  /* Sentinel */
};

static PyMemberDef cublasXt_members[] = {
  {"_CUBLAS_OP", T_OBJECT_EX, offsetof(cublasXt, _CUBLAS_OP), 0,
   "cublasOperation_t"},
  {"_CUBLAS_FILL_MODE", T_OBJECT_EX, offsetof(cublasXt, _CUBLAS_FILL_MODE), 0,
   "cublasFillMode_t"},
  {"_CUBLAS_DIAG", T_OBJECT_EX, offsetof(cublasXt, _CUBLAS_DIAG), 0,
   "cublasDiagType_t"},
  {"_CUBLAS_SIDE_MODE", T_OBJECT_EX, offsetof(cublasXt, _CUBLAS_SIDE_MODE), 0,
   "cublasSideMode_t"},
  {NULL}  /* Sentinel */
};

static PyTypeObject cublasXtType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pycublasxt.CublasXt",             /* tp_name */
  sizeof(cublasXt),                  /* tp_basicsize */
  0,                                 /* tp_itemsize */
  (destructor)cublasXt_dealloc,      /* tp_dealloc */
  0,                                 /* tp_print */
  0,                                 /* tp_getattr */
  0,                                 /* tp_setattr */
  0,                                 /* tp_reserved */
  0,                                 /* tp_repr */
  0,                                 /* tp_as_number */
  0,                                 /* tp_as_sequence */
  0,                                 /* tp_as_mapping */
  0,                                 /* tp_hash  */
  0,                                 /* tp_call */
  0,                                 /* tp_str */
  0,                                 /* tp_getattro */
  0,                                 /* tp_setattro */
  0,                                 /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                /* tp_flags */
  "Python wrapper for cublasXt",     /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  0,                         /* tp_iter */
  0,                         /* tp_iternext */
  cublasXt_methods,          /* tp_methods */
  cublasXt_members,          /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  //(initproc)cublasXt_init,      /* tp_init */
  0,                         /* tp_init */
  0,                         /* tp_alloc */
  cublasXt_new,              /* tp_new */
};


#ifndef PyMODINIT_FUNC       /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION >= 3
#define MOD_DEF(ob, name, doc, methods) \
  static struct PyModuleDef moduledef = { \
    PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
  ob = PyModule_Create(&moduledef);
#else
#define MOD_DEF(ob, name, doc, methods) \
  ob = Py_InitModule3(name, methods, doc);
#endif

static PyObject *
moduleinit(void)
{
  PyObject *m;

  if (PyType_Ready(&cublasXtType) < 0)
    return NULL;

  MOD_DEF(m, "pycublasxt",
          "Python wrapper for pycublasxt",
          pycublasxt_methods)

  Py_INCREF(&cublasXtType);
  PyModule_AddObject(m, "CublasXt", (PyObject *)&cublasXtType);
  
  if (m == NULL)
    return NULL;

  return m;
}


#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initpycublasxt(void)
{
  moduleinit();
}
#else
PyMODINIT_FUNC PyInit_pycublasxt(void)
{
  return moduleinit();
}
#endif

