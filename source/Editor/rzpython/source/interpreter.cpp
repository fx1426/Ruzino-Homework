#include <rzconsole/ConsoleObjects.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <rzpython/interpreter.hpp>
#include <rzpython/rzpython.hpp>

#include "rzconsole/string_utils.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
namespace python {

PythonInterpreter::PythonInterpreter() : python_initialized_(false)
{
    try {
        python::initialize();
        python_initialized_ = true;

        // Register Python-specific commands
        console::CommandDesc python_cmd = {
            "python",
            "Execute Python code interactively",
            [this](console::Command::Args const& args)
                -> console::Command::Result {
                if (args.size() < 2) {
                    return { false, "Usage: python <code>\n" };
                }

                std::string code;
                for (size_t i = 1; i < args.size(); ++i) {
                    if (i > 1)
                        code += " ";
                    code += args[i];
                }

                auto result = ExecutePythonCode(code);
                return { result.status, result.output };
            }
        };
        console::RegisterCommand(python_cmd);

        console::CommandDesc pyexec_cmd = {
            "exec",
            "Execute Python file",
            [this](console::Command::Args const& args)
                -> console::Command::Result {
                if (args.size() != 2) {
                    return { false, "Usage: exec <filename>\n" };
                }

                std::string code = "exec(open('" + args[1] + "').read())";
                auto result = ExecutePythonCode(code);
                return { result.status, result.output };
            }
        };
        console::RegisterCommand(pyexec_cmd);
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to initialize Python interpreter: {}", e.what());
    }
}

PythonInterpreter::~PythonInterpreter()
{
    if (python_initialized_) {
        try {
            python::finalize();
        }
        catch (...) {
            // Ignore cleanup errors
        }
    }
}

bool PythonInterpreter::ShouldHandleCommand(std::string_view command) const
{
    // Only handle as Python if interpreter is initialized
    if (!python_initialized_) {
        return false;
    }

    // Parse the command to get the first token
    auto tokens = ds::split(command);
    if (tokens.empty()) {
        return false;
    }

    std::string first_token(tokens[0]);

    // Don't handle if it's a registered console command
    if (console::FindCommand(first_token)) {
        return false;  // Let console handle it
    }

    // Don't handle our Python-specific commands (they're console commands now)
    if (first_token == "python" || first_token == "exec") {
        return false;  // Let console handle it
    }

    // Don't handle 'help' command
    if (first_token == "help") {
        return false;
    }

    // Handle everything else as Python code
    return true;
}

PythonInterpreter::Result PythonInterpreter::HandleDirectExecution(
    std::string_view cmdline)
{
    if (!python_initialized_) {
        return { false, "Python interpreter not initialized" };
    }

    return ExecutePythonCode(cmdline);
}

PythonInterpreter::Result PythonInterpreter::Execute(
    std::string_view const cmdline)
{
    // First try Python execution for Python-like code
    if (ShouldHandleCommand(cmdline)) {
        return HandleDirectExecution(cmdline);
    }

    // Fall back to base interpreter for console commands
    return console::Interpreter::Execute(cmdline);
}

std::vector<std::string> PythonInterpreter::Suggest(
    std::string_view const cmdline,
    size_t cursor_pos)
{
    if (python_initialized_ && IsPythonCode(cmdline)) {
        return SuggestPythonCompletion(cmdline);
    }

    return console::Interpreter::Suggest(cmdline, cursor_pos);
}

PythonInterpreter::Result PythonInterpreter::ExecuteCommand(
    std::string_view command,
    const std::vector<std::string>& args)
{
    // Handle Python-specific commands
    if (command == "python" || command == "exec") {
        // These are handled by registered console commands
        return { false, "Command should be handled by console system" };
    }

    return console::Interpreter::ExecuteCommand(command, args);
}

std::vector<std::string> PythonInterpreter::SuggestCommand(
    std::string_view command,
    std::string_view cmdline,
    size_t cursor_pos)
{
    if (command == "python") {
        return SuggestPythonCompletion(cmdline);
    }

    return console::Interpreter::SuggestCommand(command, cmdline, cursor_pos);
}

bool PythonInterpreter::IsValidCommand(std::string_view command) const
{
    return command == "python" || command == "exec" ||
           console::Interpreter::IsValidCommand(command);
}

bool PythonInterpreter::IsPythonCode(std::string_view code) const
{
    // This method is now mainly for suggestion purposes
    // The main detection is handled in ShouldHandleCommand
    return true;  // Simplified since detection logic moved
}

PythonInterpreter::Result PythonInterpreter::ExecutePythonCode(
    std::string_view code)
{
    if (!python_initialized_) {
        return { false, "Python interpreter not initialized\n" };
    }

    try {
        std::string code_str(code);

        // Simple and robust output capture
        python::call<void>(
            "import sys\n"
            "from io import StringIO\n"
            "_console_stdout = StringIO()\n"
            "_console_stderr = StringIO()\n"
            "_original_stdout = sys.stdout\n"
            "_original_stderr = sys.stderr\n"
            "sys.stdout = _console_stdout\n"
            "sys.stderr = _console_stderr\n");

        bool is_expression = false;
        std::string captured_output;
        std::string error_output;

        try {
            // First try as expression
            PyObject* result = PyRun_String(
                code_str.c_str(),
                Py_eval_input,
                python::main_dict,
                python::main_dict);

            if (result) {
                is_expression = true;
                // If it's not None, print the result
                if (result != Py_None) {
                    PyObject* repr_result = PyObject_Repr(result);
                    if (repr_result) {
                        const char* repr_str = PyUnicode_AsUTF8(repr_result);
                        if (repr_str) {
                            captured_output = std::string(repr_str) + "\n";
                        }
                        Py_DECREF(repr_result);
                    }
                }
                Py_DECREF(result);
            }
            else {
                // Clear the error and try as statement
                PyErr_Clear();

                PyObject* stmt_result = PyRun_String(
                    code_str.c_str(),
                    Py_file_input,
                    python::main_dict,
                    python::main_dict);

                if (stmt_result) {
                    // Statement executed successfully
                    Py_DECREF(stmt_result);
                }
                else {
                    // Get the error
                    if (PyErr_Occurred()) {
                        PyErr_Print();  // This will print to our captured
                                        // stderr
                    }
                }
            }

            // Get captured output
            try {
                std::string stdout_content =
                    python::call<std::string>("_console_stdout.getvalue()");
                std::string stderr_content =
                    python::call<std::string>("_console_stderr.getvalue()");

                // Combine stdout and stderr
                if (!stdout_content.empty()) {
                    captured_output += stdout_content;
                }
                if (!stderr_content.empty()) {
                    if (!captured_output.empty() &&
                        captured_output.back() != '\n') {
                        captured_output += "\n";
                    }
                    captured_output += stderr_content;
                }
            }
            catch (const std::exception& e) {
                captured_output +=
                    "Error getting output: " + std::string(e.what()) + "\n";
            }

            // Restore stdout/stderr
            python::call<void>(
                "sys.stdout = _original_stdout\n"
                "sys.stderr = _original_stderr\n");

            // Check if there were any errors
            bool has_error =
                !python::call<std::string>("_console_stderr.getvalue()")
                     .empty();

            return { !has_error, captured_output };
        }
        catch (const std::exception& e) {
            // Make sure to restore stdout/stderr
            try {
                python::call<void>(
                    "sys.stdout = _original_stdout\n"
                    "sys.stderr = _original_stderr\n");
            }
            catch (...) {
            }

            return { false,
                     std::string("Python execution error: ") + e.what() +
                         "\n" };
        }
    }
    catch (const std::exception& e) {
        return { false, std::string("Python error: ") + e.what() + "\n" };
    }
}

std::vector<std::string> PythonInterpreter::SuggestPythonCompletion(
    std::string_view code)
{
    if (!python_initialized_) {
        return {};
    }

    try {
        // Simple completion - get available names in global scope
        auto globals =
            python::call<std::vector<std::string>>("list(globals().keys())");

        // Filter based on current input
        std::vector<std::string> suggestions;
        std::string prefix;

        // Extract the last word as prefix
        auto pos = code.find_last_of(" \t\n()[]{}.,");
        if (pos != std::string_view::npos) {
            prefix = code.substr(pos + 1);
        }
        else {
            prefix = code;
        }

        for (const auto& name : globals) {
            if (name.size() >= prefix.size() &&
                name.substr(0, prefix.size()) == prefix) {
                suggestions.push_back(name);
            }
        }

        return suggestions;
    }
    catch (...) {
        return {};
    }
}

std::shared_ptr<console::Interpreter> CreatePythonInterpreter()
{
    return std::make_shared<PythonInterpreter>();
}

}  // namespace python

USTC_CG_NAMESPACE_CLOSE_SCOPE
