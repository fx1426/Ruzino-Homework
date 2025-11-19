#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "GUI/api.h"
#include "widget.h"
struct GLFWwindow;

USTC_CG_NAMESPACE_OPEN_SCOPE

class DockingImguiRenderer;

// Simple event system for widget communication
class GUI_API WindowEventSystem {
   public:
    using EventCallback = std::function<void(const std::string& event_data)>;
    
    void subscribe(const std::string& event_name, EventCallback callback);
    void emit(const std::string& event_name, const std::string& event_data = "");
    
   private:
    std::unordered_map<std::string, std::vector<EventCallback>> subscribers_;
};

// Represents a window in a GUI application, providing basic functionalities
// such as initialization and rendering.
class GUI_API Window {
   public:
    // Constructor that sets the window's title.
    explicit Window();

    virtual ~Window();

    // Enters the main rendering loop.
    float get_elapsed_time();
    void run();
    void register_widget(std::unique_ptr<IWidget> unique);

    void register_function_before_frame(
        const std::function<void(Window *)> &callback);
    void register_function_after_frame(
        const std::function<void(Window *)> &callback);

    void register_openable_widget(
        std::unique_ptr<IWidgetFactory> window_factory,
        const std::vector<std::string> &menu_item);
    IWidget *get_widget(const std::string &unique_name) const;
    std::vector<IWidget *> get_widgets() const;
    
    // Event system access
    WindowEventSystem& events() { return event_system_; }

    void close();

    int get_size_x() const;
    int get_size_y() const;

    virtual void SetFullscreen(bool enabled);
    [[nodiscard]] bool IsFullscreen() const;

    virtual void SetMaximized(bool enabled);
    [[nodiscard]] bool IsMaximized() const;

   protected:
    std::unique_ptr<DockingImguiRenderer> imguiRenderPass;
    float elapsedTimeSeconds = 0.0f;
    WindowEventSystem event_system_;
    friend class DockingImguiRenderer;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
