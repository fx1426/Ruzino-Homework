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

#include <rzconsole/ConsoleInterpreter.h>
#include <rzconsole/ConsoleObjects.h>
#include <rzconsole/imgui_console.h>
#include <rzconsole/spdlog_console_sink.h>
#include <rzconsole/string_utils.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstring>
#include <fstream>

RUZINO_NAMESPACE_OPEN_SCOPE

static ImVec4 getSeverityColor(LogSeverity severity)
{
    switch (severity) {
        case LogSeverity::Info: return ImVec4(.6f, .8f, 1.f, 1.f);
        case LogSeverity::Warning: return ImVec4(1.f, .5f, 0.f, 1.f);
        case LogSeverity::Error: return ImVec4(1.f, 0.f, 0.f, 1.f);
        default: break;
    }
    return ImVec4(1.f, 1.f, 1.f, 1.f);
}

ImGui_Console::ImGui_Console(
    std::shared_ptr<console::Interpreter> interpreter,
    Options const& options)
    : m_Options(options),
      m_Interpreter(interpreter)
{
    // Only setup spdlog integration if capture_log is enabled
    if (options.capture_log) {
        setup_console_logging(this);
    }

    // Load history from disk
    LoadHistory();
}
ImGui_Console::~ImGui_Console()
{
    // Save history to disk before destruction
    SaveHistory();

    // Must disconnect the sink from spdlog BEFORE destroying this object
    // Otherwise spdlog will try to log to a deleted console object
    if (m_Options.capture_log) {
        auto sink = get_global_console_sink();
        sink->set_console(nullptr);

        // Remove the console logger as default logger if it exists
        try {
            spdlog::set_default_logger(nullptr);
        }
        catch (...) {
            // Ignore exceptions during cleanup
        }
    }
}

void ImGui_Console::Print(char const* fmt, ...)
{
    InputBuffer buf;
    std::va_list args;

    va_start(args, fmt);
    vsnprintf(buf.data(), buf.size(), fmt, args);
    buf.back() = 0;
    va_end(args);

    LogItem item;
    item.text = buf.data();
    m_ItemsLog.push_back(item);
}

void ImGui_Console::Print(std::string_view line)
{
    LogItem item;
    item.text = line;
    m_ItemsLog.push_back(item);
}

void ImGui_Console::PrintWithSeverity(const char* text, LogSeverity severity)
{
    LogItem item;
    item.severity = severity;
    item.textColor = getSeverityColor(severity);
    item.text = text;
    m_ItemsLog.push_back(item);
}

void ImGui_Console::ClearLog()
{
    m_ItemsLog.clear();
}

void ImGui_Console::ClearHistory()
{
    m_History.clear();
    m_HistoryPos = -1;
}

bool ImGui_Console::BuildUI()
{
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Close Console")) {
            // For IWidget, we don't directly control closing, return false to
            // indicate widget should close
        }
        ImGui::EndPopup();
    }

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Edit")) {
            bool clearLog = ImGui::MenuItem("Clear Log");
            bool clearHistory = ImGui::MenuItem("Clear History");
            bool clearAll = ImGui::MenuItem("Clear All");

            if (clearLog || clearAll)
                this->ClearLog();
            if (clearHistory || clearAll)
                this->ClearHistory();
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // ImGui::Separator();

    // Log area

    const float footer_height =
        ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
    ImGui::BeginChild(
        "Log panel",
        ImVec2(0, -footer_height),
        false,
        ImGuiWindowFlags_HorizontalScrollbar);

    // right click popup on log panel
    if (ImGui::BeginPopupContextWindow()) {
        if (ImGui::Selectable("Clear"))
            ClearLog();
        ImGui::EndPopup();
    }

    ImGui::PushStyleVar(
        ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));  // Tighten spacing

    for (auto const& item : m_ItemsLog) {
        bool showItem = true;
        switch (item.severity) {
            case LogSeverity::Info: showItem = m_Options.show_info; break;
            case LogSeverity::Warning:
                showItem = m_Options.show_warnings;
                break;
            case LogSeverity::Error: showItem = m_Options.show_errors; break;
            default: break;
        }

        if (showItem) {
            ImGui::PushStyleColor(ImGuiCol_Text, item.textColor);
            ImGui::TextUnformatted(item.text.c_str());
            ImGui::PopStyleColor();
        }
    }

    if (m_Options.scroll_to_bottom ||
        (m_Options.auto_scroll &&
         ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
        ImGui::SetScrollHereY(1.f);
    }

    m_Options.scroll_to_bottom = false;
    ImGui::PopStyleVar();
    ImGui::EndChild();  // end log scroll area

    ImGui::Separator();

    // Command line
    bool reclaim_focus = false;
    auto flags = ImGuiInputTextFlags_EnterReturnsTrue |
                 ImGuiInputTextFlags_CallbackCompletion |
                 ImGuiInputTextFlags_CallbackHistory;
    if (ImGui::InputText(
            "##consoleInput",
            m_InputBuffer.data(),
            m_InputBuffer.size(),
            flags,
            [](ImGuiInputTextCallbackData* data) {
                ImGui_Console* console = (ImGui_Console*)data->UserData;
                return console->TextEditCallback(data);
            },
            (void*)this)) {
        if (m_InputBuffer.front() != '0') {
            // Check if we're executing from history
            std::string currentInput(m_InputBuffer.data());
            m_ExecutingFromHistory =
                (m_HistoryPos >= 0 && m_HistoryPos < m_History.size() &&
                 m_History[m_HistoryPos] == currentInput);

            this->ExecCommand(m_InputBuffer.data());
            m_InputBuffer.front() = 0;
        }
        reclaim_focus = true;
    }

    // Auto-focus on window apparition
    ImGui::SetItemDefaultFocus();
    if (reclaim_focus)
        ImGui::SetKeyboardFocusHere(-1);  // Auto focus previous widget

    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Filters : ");
    ImGui::SameLine();
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1);
    auto filterButton =
        [](char const* label, bool* value, LogSeverity severity) {
            ImGui::PushStyleColor(ImGuiCol_Border, getSeverityColor(severity));
            ImGui::Checkbox(label, value);
            ImGui::PopStyleColor();
        };
    filterButton("Errors", &m_Options.show_errors, LogSeverity::Error);
    ImGui::SameLine();
    filterButton("Warnings", &m_Options.show_warnings, LogSeverity::Warning);
    ImGui::SameLine();
    filterButton("Info", &m_Options.show_info, LogSeverity::Info);
    ImGui::PopStyleVar();  // FrameBorder
    return true;
}
std::string ImGui_Console::GetWindowUniqueName()
{
    return "Python Console";
}

static void printLines(ImGui_Console& console, std::string const& output)
{
    if (output.empty())
        return;

    std::string line;
    for (int start = 0, curr = 0; curr < (int)output.size(); ++curr) {
        if ((output[curr] == '\r') || (output[curr] == '\n')) {
            console.Print(std::string_view(&output[start], curr - start));
            start = ++curr;
        }
    }
}

void ImGui_Console::ExecCommand(char const* cmdline)
{
    std::string_view const cmd = cmdline;
    if (!cmd.empty()) {
        this->Print("> %s", cmd.data());

        auto result = m_Interpreter->Execute(cmd);

        // Always show something - even if execution failed
        if (!result.output.empty()) {
            printLines(*this, result.output);
        }
        else if (!result.status) {
            // Failure - show error message if output is empty
            this->Print("Command failed with no output");
        }

        // Add to history regardless of success/failure
        std::string cmdStr(cmd.data());

        if (m_ExecutingFromHistory) {
            // Command came from history - just move it to the top
            auto it = std::find(m_History.begin(), m_History.end(), cmdStr);
            if (it != m_History.end()) {
                m_History.erase(it);
                m_History.push_back(cmdStr);
            }
        }
        else {
            // New command - remove if it exists and add to end
            auto it = std::find(m_History.begin(), m_History.end(), cmdStr);
            if (it != m_History.end()) {
                m_History.erase(it);
            }
            m_History.push_back(cmdStr);
        }

        m_HistoryPos = -1;
        m_ExecutingFromHistory = false;
    }
}

// XXXX mk: we should probably use the columns features instead ?
static void printColumns(
    ImGui_Console& console,
    std::vector<std::string> const& items)
{
    auto computeLineWidth = []() {
        // XXXX mk: this only works if the font is monospace !
        float width = ImGui::CalcItemWidth();
        ImVec2 charWidth = ImGui::CalcTextSize("A");
        return (size_t)(width / charWidth.x);
    };

    size_t max_len = 0;
    for (auto const& candidate : items)
        max_len = std::max(max_len, candidate.size());

    size_t line_width = computeLineWidth();
    size_t ncolumns = line_width / max_len;

    std::string line;
    int col = 1;
    for (auto const& candidate : items) {
        if ((col % ncolumns) != 0) {
            line += candidate;
            line += ' ';
            ++col;
        }
        else {
            console.Print(line.c_str());
            line.clear();
            col = 1;
        }
    }
    if (!line.empty())
        console.Print(line.c_str());
}

static std::string extendKeyword(
    std::string_view keyword,
    std::vector<std::string> const& candidates)
{
    std::string match(keyword.data(), keyword.length());
    while (true) {
        int c = -1, cpos = (int)match.size();
        for (std::string_view const candidate : candidates) {
            if (cpos < candidate.size()) {
                if (c == -1)
                    c = candidate[cpos];
                else if (c != candidate[cpos])
                    return match;
            }
            else
                return match;
        }
        match.push_back(c);
    }
}

inline std::string_view isolateKeyword(std::string_view line)
{
    ds::ltrim(line);
    if (auto it = std::find_if(
            line.rbegin(),
            line.rend(),
            [](int ch) { return std::isspace(ch); });
        it != line.rend()) {
        line.remove_prefix(std::distance(line.begin(), it.base()));
    }
    return line;
}

// Get the incomplete part after the last '.' or space
inline std::string_view getIncompletePart(std::string_view line)
{
    ds::ltrim(line);

    // Find the last '.' or space
    size_t lastSep = line.find_last_of(". \t");
    if (lastSep != std::string_view::npos) {
        return line.substr(lastSep + 1);
    }

    // No separator found, return the whole line
    return line;
}

int ImGui_Console::AutoCompletionCallback(ImGuiInputTextCallbackData* data)
{
    std::string_view cmdline(data->Buf, data->CursorPos);
    std::string_view keyword = isolateKeyword(cmdline);
    std::string_view incompletePart = getIncompletePart(cmdline);

    // Check if we're cycling through previous candidates
    bool isCycling = false;
    if (!m_CompletionCandidates.empty() && !incompletePart.empty()) {
        std::string incomplete(incompletePart);
        // Check if current incomplete part matches one of the previous
        // candidates
        for (const auto& candidate : m_CompletionCandidates) {
            if (incomplete == candidate) {
                isCycling = true;
                break;
            }
        }
    }

    if (isCycling) {
        // Cycle to next candidate (forward)
        m_CompletionIndex =
            (m_CompletionIndex + 1) % m_CompletionCandidates.size();

        // Replace incomplete part with next candidate
        data->DeleteChars(
            data->CursorPos - incompletePart.size(), incompletePart.size());
        data->InsertChars(
            data->CursorPos, m_CompletionCandidates[m_CompletionIndex].c_str());
    }
    else {
        // New completion request
        auto candidates = m_Interpreter->Suggest(data->Buf, data->CursorPos);

        if (candidates.empty()) {
            // No candidates - clear state and return
            m_CompletionCandidates.clear();
            m_CompletionIndex = 0;
            return 0;
        }

        // Store candidates for cycling
        m_CompletionCandidates = candidates;
        m_CompletionIndex = 0;

        // Replace incomplete part with first candidate
        data->DeleteChars(
            data->CursorPos - incompletePart.size(), incompletePart.size());
        data->InsertChars(data->CursorPos, candidates[0].c_str());

        // Show all options if multiple candidates
        if (candidates.size() > 1) {
            if (candidates.size() < 64)
                printColumns(*this, candidates);
            else
                Print("Too many matches (%d)", (int)candidates.size());
        }
        else {
            // Single match - clear state without adding space
            m_CompletionCandidates.clear();
        }
    }
    return 0;
}

int ImGui_Console::AutoCompletionCallbackReverse(
    ImGuiInputTextCallbackData* data)
{
    std::string_view cmdline(data->Buf, data->CursorPos);
    std::string_view incompletePart = getIncompletePart(cmdline);

    // Check if we're cycling through previous candidates
    if (m_CompletionCandidates.empty()) {
        return 0;  // No candidates to cycle through
    }

    bool isCycling = false;
    if (!incompletePart.empty()) {
        std::string incomplete(incompletePart);
        for (const auto& candidate : m_CompletionCandidates) {
            if (incomplete == candidate) {
                isCycling = true;
                break;
            }
        }
    }

    if (isCycling) {
        // Cycle to previous candidate (reverse direction)
        if (m_CompletionIndex == 0) {
            m_CompletionIndex = m_CompletionCandidates.size() - 1;
        }
        else {
            m_CompletionIndex--;
        }

        // Replace incomplete part with previous candidate
        data->DeleteChars(
            data->CursorPos - incompletePart.size(), incompletePart.size());
        data->InsertChars(
            data->CursorPos, m_CompletionCandidates[m_CompletionIndex].c_str());
    }

    return 0;
}

int ImGui_Console::HistoryKeyCallback(ImGuiInputTextCallbackData* data)
{
    int prev_history_pos = m_HistoryPos;
    switch (data->EventKey) {
        case ImGuiKey_UpArrow:
            if (m_HistoryPos == -1)
                m_HistoryPos = m_History.size() - 1;
            else if (m_HistoryPos > 0)
                m_HistoryPos--;
            break;
        case ImGuiKey_DownArrow:
            if (m_HistoryPos != -1) {
                ++m_HistoryPos;
                if (m_HistoryPos >= (int)m_History.size())
                    m_HistoryPos = -1;
            }
            break;
        default: break;
    }

    if (prev_history_pos != m_HistoryPos) {
        const char* history_str =
            (m_HistoryPos >= 0) ? m_History[m_HistoryPos].c_str() : "";
        snprintf(data->Buf, data->BufSize, "%s", history_str);
        data->BufTextLen = (int)strlen(history_str);
        data->BufDirty = true;
        data->CursorPos = data->BufTextLen;  // Move cursor to end
        data->SelectionStart = data->SelectionEnd = data->CursorPos;
    }
    return 0;
}

int ImGui_Console::TextEditCallback(ImGuiInputTextCallbackData* data)
{
    switch (data->EventFlag) {
        case ImGuiInputTextFlags_CallbackCompletion:
            // Normal Tab (without Shift)
            return AutoCompletionCallback(data);

        case ImGuiInputTextFlags_CallbackHistory:
            return HistoryKeyCallback(data);
    }

    return 0;
}

void ImGui_Console::SetLogCapture(bool enable)
{
    if (enable && !m_Options.capture_log) {
        // Enable logging capture
        setup_console_logging(this);
        m_Options.capture_log = true;
    }
    else if (!enable && m_Options.capture_log) {
        // Disable logging capture by setting console to nullptr
        auto sink = get_global_console_sink();
        sink->set_console(nullptr);
        m_Options.capture_log = false;
    }
}

void ImGui_Console::LoadHistory()
{
    std::ifstream file(".ruzino_console_history");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                m_History.push_back(line);
            }
        }
        file.close();
    }
}

void ImGui_Console::SaveHistory()
{
    std::ofstream file(".ruzino_console_history");
    if (file.is_open()) {
        // Save up to 1000 most recent commands
        size_t start = m_History.size() > 1000 ? m_History.size() - 1000 : 0;
        for (size_t i = start; i < m_History.size(); ++i) {
            file << m_History[i] << '\n';
        }
        file.close();
    }
}

RUZINO_NAMESPACE_CLOSE_SCOPE
