/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
License for Dear ImGui

Copyright (c) 2014-2019 Omar Cornut

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <GUI/widget.h>
#include <imgui.h>
#include <spdlog/fwd.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace console {
class Interpreter;
}

enum class LogSeverity { None = 0, Info, Warning, Error };

class RZCONSOLE_API ImGui_Console : public IWidget {
   public:
    struct Options {
        // std::shared_ptr<RegisteredFont> font;        // it is recommended to
        // specify a monospace font

        bool auto_scroll = true;  // automatically keep log output scrolled to
                                  // the most recent item
        bool scroll_to_bottom = false;  // scoll to botom on console creation,
                                        // if the log is not empty

        bool capture_log =
            true;  // captures spdlog event logs & redirects to the console
        bool show_info = false;  // default state of log events filters
        bool show_warnings = true;
        bool show_errors = true;
    };

    ImGui_Console(
        std::shared_ptr<console::Interpreter> interpreter,
        Options const& opts);

    ~ImGui_Console();

    void Print(char const* fmt, ...);

    void Print(std::string_view line);

    void PrintWithSeverity(const char* text, LogSeverity severity);

    void ClearLog();

    void ClearHistory();

    // Enable or disable spdlog capture
    void SetLogCapture(bool enable);

    // Override IWidget methods
    bool BuildUI() override;

   protected:
    // Override keyboard handling to ensure Shift+Tab works
    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        // Don't intercept Tab key - let ImGui handle it
        return false;
    }

   protected:
    const char* GetWindowName() override
    {
        return "Console";
    }
    std::string GetWindowUniqueName() override;
    // Menu bar support for console
    bool HasMenuBar() const override
    {
        return true;
    }

   private:
    int HistoryKeyCallback(ImGuiInputTextCallbackData* data);

    int AutoCompletionCallback(ImGuiInputTextCallbackData* data);

    int AutoCompletionCallbackReverse(ImGuiInputTextCallbackData* data);

    int TextEditCallback(ImGuiInputTextCallbackData* data);

    void ExecCommand(char const* cmd);

    void LoadHistory();
    void SaveHistory();

   private:
    typedef std::array<char, 256> InputBuffer;
    InputBuffer m_InputBuffer = { 0 };

    struct LogItem {
        LogSeverity severity = LogSeverity::None;
        ImVec4 textColor = ImVec4(1.f, 1.f, 1.f, 1.f);
        std::string text;
    };

    std::vector<LogItem> m_ItemsLog;
    std::vector<std::string> m_History;
    int m_HistoryPos = -1;
    bool m_ExecutingFromHistory = false;

    // Tab completion cycling state
    std::vector<std::string> m_CompletionCandidates;
    size_t m_CompletionIndex = 0;

   private:
    Options m_Options;

    std::shared_ptr<console::Interpreter> m_Interpreter;
};

RUZINO_NAMESPACE_CLOSE_SCOPE
