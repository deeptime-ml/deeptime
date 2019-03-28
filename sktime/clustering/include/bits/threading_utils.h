//
// Created by clonkscher on 8/8/17.
//

#ifndef PYEMMA_THREADING_UTILS_H
#define PYEMMA_THREADING_UTILS_H


#include <stdexcept>
#include <thread>
#include <utility>

/**
 * scoped_thread implementation
 */
class scoped_thread {
    std::thread t;
public:
    /**
     * Creates a new scoped_thread based on a thread object
     * @param _t the reference thread
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
    scoped_thread(scoped_thread &&rhs) {
        t = std::move(rhs.t);
    }

    /**
     * moves another scoped_thread into this
     * @param rhs the other scoped_thread
     * @return myself
     */
    scoped_thread &operator=(scoped_thread &&rhs) {
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


#endif //PYEMMA_THREADING_UTILS_H
