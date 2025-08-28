#include <GUI/window.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <rzpython/rzpython.hpp>
#include <stdexcept>
#include <unordered_map>

namespace nb = nanobind;

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace python {

// Global variables - accessible from template implementations
PyObject* main_module = nullptr;
PyObject* main_dict = nullptr;
bool initialized = false;
std::unordered_map<std::string, nb::object> bound_objects;

void initialize()
{
    if (initialized) {
        return;
    }

    // Add path to ensure Python finds our modules
    Py_Initialize();
    if (!Py_IsInitialized()) {
        throw std::runtime_error("Failed to initialize Python interpreter");
    }

    // Simple initialization marker without problematic import hooks
    try {
        PyRun_SimpleString(
            "import sys\n"
            "sys._rzpython_initialized = True\n");
    }
    catch (...) {
        // Ignore setup errors
    }

    main_module = PyImport_AddModule("__main__");
    if (!main_module) {
        throw std::runtime_error("Failed to get __main__ module");
    }

    main_dict = PyModule_GetDict(main_module);
    if (!main_dict) {
        throw std::runtime_error("Failed to get __main__ dictionary");
    }

    initialized = true;
}

void finalize()
{
    if (!initialized) {
        return;
    }

    // Clear our bound objects
    try {
        for (const auto& pair : bound_objects) {
            PyDict_DelItemString(main_dict, pair.first.c_str());
        }
        bound_objects.clear();
    }
    catch (...) {
        // Ignore cleanup errors
    }

    // Reset main module references
    main_module = nullptr;
    main_dict = nullptr;

    // Finalize Python interpreter
    Py_Finalize();

    initialized = false;
}

void import(const std::string& module_name)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* module = PyImport_ImportModule(module_name.c_str());
    if (!module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import module: " + module_name);
    }

    // Add module to main dict so it can be accessed
    PyDict_SetItemString(main_dict, module_name.c_str(), module);
    Py_DECREF(module);
}

// Internal helper for raw Python object return
PyObject* call_raw(const std::string& code)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* result =
        PyRun_String(code.c_str(), Py_eval_input, main_dict, main_dict);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error("Failed to execute Python code: " + code);
    }

    return result;  // Caller is responsible for DECREF
}

// Only keep specializations for void (which needs different PyRun_String mode)
// and primitive types that need special handling
template<>
void call<void>(const std::string& code)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    PyObject* result =
        PyRun_String(code.c_str(), Py_file_input, main_dict, main_dict);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error("Failed to execute Python code: " + code);
    }

    Py_DECREF(result);
}

}  // namespace python

USTC_CG_NAMESPACE_CLOSE_SCOPE
