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

#include <rzconsole/ConsoleObjects.h>
#include <rzconsole/string_utils.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>

USTC_CG_NAMESPACE_OPEN_SCOPE
namespace console {

using int2 = glm::ivec2;
using int3 = glm::ivec3;
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;

static std::string const emptyString;

// helper : compile regex safely & catch user errors
inline std::optional<std::regex> regex_from_char(char const* s)
{
    std::regex rx;
    if (s) {
        try {
            rx = s;
        }
        catch (std::regex_error const& err) {
            spdlog::error(err.what());
            return std::nullopt;
        }
    }
    return rx;
}

//
// Console Object Dictionary
//

class ObjectDictionary {
   public:
    static inline bool IsValidName(char const* name)
    {
        return (name && (std::strlen(name) > 0)) ? true : false;
    }

    Object* RegisterCommand(CommandDesc const& desc)
    {
        if (IsValidName(desc.name)) {
            if (!desc.on_execute) {
                spdlog::critical(
                    "attempting to register console command '{}' with no "
                    "execution function",
                    desc.name);
                std::abort();
            }
            std::lock_guard<std::mutex> lock(m_Mutex);
            if (auto it = m_Dictionary.find(desc.name);
                it == m_Dictionary.end()) {
                auto* cmd = new Command(
                    desc.description, desc.on_execute, desc.on_suggest);
                m_Dictionary[desc.name] = cmd;
                return cmd;
            }
            else
                spdlog::error(
                    "console command with name '{}' already exists", desc.name);
        }
        else
            spdlog::error(
                "attempting to register a console command with invalid name "
                "'{}'",
                desc.name);
        return nullptr;
    }

    bool UnregisterCommand(std::string_view name)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (auto it = m_Dictionary.find(name); it != m_Dictionary.end()) {
            if (it->second->AsCommand()) {
                m_Dictionary.erase(it);
                return true;
            }
            else
                spdlog::error(
                    "unregister command '{}'; object is not a console command",
                    std::string(name));
        }
        else
            spdlog::error(
                "unregister command '{}'; command does not exist",
                std::string(name));
        return false;
    }

    Object* FindObject(std::string_view name)
    {
        if (!name.empty()) {
            std::lock_guard<std::mutex> lock(m_Mutex);
            auto it = m_Dictionary.find(name);
            if (it != m_Dictionary.end())
                return it->second;
        }
        return nullptr;
    }

    std::vector<std::string_view> FindObjectNames(char const* regex)
    {
        std::vector<std::string_view> matches;
        if (auto rx = regex_from_char(regex)) {
            for (auto& it : m_Dictionary)
                if (std::regex_match(it.first, *rx))
                    matches.push_back(std::string_view(it.first));
        }
        return matches;
    }

    std::vector<Object*> FindObjects(char const* regex)
    {
        std::vector<Object*> matches;
        if (auto rx = regex_from_char(regex)) {
            for (auto& it : m_Dictionary)
                if (std::regex_match(it.first, *rx))
                    matches.push_back(it.second);
        }
        return matches;
    }

    std::string const& GetObjectName(Object const* cobj)
    {
        // slow linear search under the assumption that this is only called very
        // rarely
        if (cobj) {
            std::lock_guard<std::mutex> lock(m_Mutex);
            for (auto const& it : m_Dictionary) {
                if (it.second == cobj)
                    return it.first;
            }
            spdlog::error("unregistered object");
        }
        return emptyString;
    }

    void Reset()
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Dictionary.clear();
    }

   private:
    // note: the dictionary deliberately leaks its ConsoleObjects* in order to
    // guarantee that any reference to the memory will still be valid when the
    // application shuts down and implicit destructors are invoked. The
    // "correct" implementation would to own lifespan with shared/weak_ptr, but
    // this adds a lot of atomic & error checking burdens which were not deemed
    // to be worth it.
    std::mutex m_Mutex;
    std::map<std::string, Object*, std::less<>> m_Dictionary;

} objectsDictionary;

//
// Implementation
//

//
// Console Object
//

std::string const& Object::GetName() const
{
    return objectsDictionary.GetObjectName(this);
}

//
// Console Command
//

Command::Command(
    char const* description,
    OnExecuteFunction onexec,
    OnSuggestFunction onsuggest)
    : Object(description),
      m_OnExecute(onexec),
      m_OnSuggest(onsuggest)
{
}

Command::Result Command::Execute(Args const& args)
{
    if (m_OnExecute)
        return m_OnExecute(args);
    else
        spdlog::error("console command '{}' has no function", this->GetName());
    return Result();
}

std::vector<std::string> Command::Suggest(
    std::string_view cmdline,
    size_t cursor_pos)
{
    if (m_OnSuggest)
        return m_OnSuggest(cmdline, cursor_pos);
    else
        return {};
}

//
// Dictionary implementation
//

bool RegisterCommand(CommandDesc const& desc)
{
    return objectsDictionary.RegisterCommand(desc) != nullptr;
}

bool UnregisterCommand(std::string_view name)
{
    return objectsDictionary.UnregisterCommand(name);
}

Object* FindObject(std::string_view name)
{
    return objectsDictionary.FindObject(name);
}

std::vector<std::string_view> MatchObjectNames(char const* regex)
{
    return objectsDictionary.FindObjectNames(regex);
}

std::vector<Object*> MatchObjects(char const* regex)
{
    return objectsDictionary.FindObjects(regex);
}

Command* FindCommand(std::string_view name)
{
    if (Object* cobj = FindObject(name))
        return cobj->AsCommand();
    return nullptr;
}

void ResetAll()
{
    objectsDictionary.Reset();
}

}  // end namespace console
USTC_CG_NAMESPACE_CLOSE_SCOPE
