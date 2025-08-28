#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <string>

#include "api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace python {

// Type aliases for common ndarray configurations
using numpy_array_f32 = nanobind::ndarray<nanobind::numpy, float>;
using numpy_array_f64 = nanobind::ndarray<nanobind::numpy, double>;
using numpy_array_i32 = nanobind::ndarray<nanobind::numpy, int32_t>;
using numpy_array_i64 = nanobind::ndarray<nanobind::numpy, int64_t>;

using torch_tensor_f32 = nanobind::ndarray<nanobind::pytorch, float>;
using torch_tensor_f64 = nanobind::ndarray<nanobind::pytorch, double>;
using torch_tensor_i32 = nanobind::ndarray<nanobind::pytorch, int32_t>;
using torch_tensor_i64 = nanobind::ndarray<nanobind::pytorch, int64_t>;

// CPU ndarray types
using cpu_array_f32 = nanobind::ndarray<float, nanobind::device::cpu>;
using cpu_array_f64 = nanobind::ndarray<double, nanobind::device::cpu>;
using cpu_array_i32 = nanobind::ndarray<int32_t, nanobind::device::cpu>;
using cpu_array_i64 = nanobind::ndarray<int64_t, nanobind::device::cpu>;

// CUDA ndarray types for GPU memory
using cuda_array_f32 = nanobind::ndarray<float, nanobind::device::cuda>;
using cuda_array_f64 = nanobind::ndarray<double, nanobind::device::cuda>;
using cuda_array_i32 = nanobind::ndarray<int32_t, nanobind::device::cuda>;
using cuda_array_i64 = nanobind::ndarray<int64_t, nanobind::device::cuda>;

// Initialize Python interpreter
RZPYTHON_API void initialize();

// Finalize Python interpreter
RZPYTHON_API void finalize();

// Import a Python module
RZPYTHON_API void import(const std::string& module_name);

// Internal helper for dynamic type conversion
RZPYTHON_API PyObject* call_raw(const std::string& code);

// Generic call function - implemented in header for template instantiation
template<typename T>
T call(const std::string& code);

// Only void needs special handling (different PyRun_String mode)
template<>
RZPYTHON_API void call<void>(const std::string& code);

// Bind C++ object to Python variable name
template<typename T>
void bind_object(const std::string& name, T* obj);

// Helper function to get nanobind cast for objects
template<typename T>
void reference(const std::string& name, T* obj);

// Send C++ data to Python variable (by value)
template<typename T>
void send(const std::string& name, const T& value);

}  // namespace python

USTC_CG_NAMESPACE_CLOSE_SCOPE

// Template implementations - include after declarations
#include "rzpython_impl.hpp"
