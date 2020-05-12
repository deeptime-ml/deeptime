#pragma once

#include <thread>

namespace sktime {
namespace thread {

class scoped_thread {
    std::thread t;
public:
    /**
     * Creates a new scoped_thread
     * @tparam Function the function type that is executed by the encapsulated thread
     * @tparam Args argument types to the function
     * @param fun the function instance that is executed by the encapsulated thread
     * @param args arguments to that function
     */
    template<typename Function, typename... Args>
    explicit scoped_thread(Function &&fun, Args &&... args)
            : t(std::forward<Function>(fun), std::forward<Args>(args)...) {
        if (!t.joinable()) throw std::logic_error("No thread!");
    }

    /**
     * joins the contained thread if its joinable
     */
    ~scoped_thread() {
        if (t.joinable()) {
            t.join();
        }
    }

    /**
     * moves another scoped_thread into this
     * @param rhs the other scoped_thread
     */
    scoped_thread(scoped_thread &&rhs) noexcept {
        t = std::move(rhs.t);
    }

    /**
     * moves another scoped_thread into this
     * @param rhs the other scoped_thread
     * @return myself
     */
    scoped_thread &operator=(scoped_thread &&rhs) noexcept {
        t = std::move(rhs.t);
        return *this;
    }

    /**
     * copying is not allowed
     */
    scoped_thread(const scoped_thread &) = delete;

    /**
     * copying is not allowed
     */
    scoped_thread &operator=(const scoped_thread &) = delete;
};

}
}
