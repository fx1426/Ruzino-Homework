#pragma once
#include "RHI/internal/map.h"

#define INTEGER_LIST                                                          \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,   \
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,   \
        55, 56, 57, 58, 59, 60, 61, 62

template<int N>
class SocketTypePlaceHolder { };
#define INSTANTIATE_SOCKET_TYPE_PLACEHOLDER(N) \
    template class SocketTypePlaceHolder<N>;

MACRO_MAP(INSTANTIATE_SOCKET_TYPE_PLACEHOLDER, INTEGER_LIST)

RUZINO_NAMESPACE_OPEN_SCOPE
// We have _uniqueSocketType in the MaterialXNodeTree class
Ruzino::SocketType MaterialXNodeTree::get_unique_socket_type(const char* name)
{
    auto type = entt::resolve(get_entt_ctx(), entt::hashed_string{ name });
    if (type) {
        return type;
    }

    // Get the next socket type and increment counter
    SocketType result;

    switch (_uniqueSocketType++) {
#define CASE(N)                                              \
    case N:                                                  \
        entt::meta<SocketTypePlaceHolder<N>>(get_entt_ctx()) \
            .type(entt::hashed_string(name));                \
        break;

        MACRO_MAP(CASE, INTEGER_LIST)
        default:
            // Handle error or expand as needed
            assert(
                _uniqueSocketType <= 10 &&
                "Out of pre-instantiated socket types");
            result = {};  // Return empty/invalid socket type
    }

    return get_socket_type(entt::hashed_string{ name });
}

RUZINO_NAMESPACE_CLOSE_SCOPE