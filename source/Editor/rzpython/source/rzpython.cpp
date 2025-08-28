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

    // Setup USD DLL path for Windows to resolve Boost.Python import issues
    try {
        PyRun_SimpleString(
            "import os\n"
            "import sys\n"
            "# Setup PXR_USD_WINDOWS_DLL_PATH for USD imports\n"
            "current_dir = os.getcwd()\n"
            "pxr_dll_path = os.environ.get('PXR_USD_WINDOWS_DLL_PATH', '')\n"
            "if current_dir not in pxr_dll_path:\n"
            "    if pxr_dll_path:\n"
            "        os.environ['PXR_USD_WINDOWS_DLL_PATH'] = current_dir + "
            "os.pathsep + pxr_dll_path\n"
            "    else:\n"
            "        os.environ['PXR_USD_WINDOWS_DLL_PATH'] = current_dir\n"
            "# Also add to system PATH as backup\n"
            "system_path = os.environ.get('PATH', '')\n"
            "if current_dir not in system_path:\n"
            "    os.environ['PATH'] = current_dir + os.pathsep + system_path\n"
            "# Add ./python to sys.path for module imports\n"
            "python_path = os.path.join(current_dir, 'python')\n"
            "if python_path not in sys.path:\n"
            "    sys.path.append(python_path)\n"
            "print(f'USD DLL Path setup: "
            "{os.environ.get(\"PXR_USD_WINDOWS_DLL_PATH\", \"Not set\")}')\n");
    }
    catch (...) {
        // Ignore USD setup errors - USD might not be available
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

    import("GUI_py");
}

void finalize()
{
    if (!initialized) {
        return;
    }

    // Clear our bound objects
    try {
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

bool is_boost_python_module(const std::string& module_name)
{
    // Check if this is a known Boost.Python module
    static const std::vector<std::string> boost_modules = {
        "pxr", "Vt", "Gf", "Tf", "Sdf", "Usd"
    };

    for (const auto& boost_mod : boost_modules) {
        if (module_name.find(boost_mod) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void safe_import(const std::string& module_name)
{
    if (!initialized) {
        throw std::runtime_error("Python interpreter not initialized");
    }

    try {
        // For Boost.Python modules, we need special handling
        if (is_boost_python_module(module_name)) {
            // Try to import with error suppression and compatibility mode
            std::string safe_code =
                "try:\n"
                "    import " +
                module_name +
                "\n"
                "    _import_success = True\n"
                "except Exception as e:\n"
                "    print(f'Warning: Failed to import " +
                module_name +
                ": {e}')\n"
                "    _import_success = False\n";

            PyObject* result = PyRun_String(
                safe_code.c_str(), Py_file_input, main_dict, main_dict);
            if (!result) {
                PyErr_Print();
                throw std::runtime_error(
                    "Failed to safely import module: " + module_name);
            }
            Py_DECREF(result);

            // Check if import was successful
            PyObject* success =
                PyDict_GetItemString(main_dict, "_import_success");
            if (!success || !PyObject_IsTrue(success)) {
                throw std::runtime_error(
                    "Module import failed: " + module_name);
            }
        }
        else {
            // Use regular import for non-Boost.Python modules
            import(module_name);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            "Safe import failed for " + module_name + ": " + e.what());
    }
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
