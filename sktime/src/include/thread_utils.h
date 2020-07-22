/********************************************************************************
 * This file is part of scikit-time.                                            *
 *                                                                              *
 * Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)         *
 *                                                                              *
 * scikit-time is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU Lesser General Public License as published by  *
 * the Free Software Foundation, either version 3 of the License, or            *
 * (at your option) any later version.                                          *
 *                                                                              *
 * This program is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
 * GNU General Public License for more details.                                 *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
 ********************************************************************************/

#pragma once

#include <thread>
#include <atomic>

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

template<typename T>
class copyable_atomic {
    std::atomic<T> _a;
public:
    /**
     * the contained value
     */
    using value_type = T;

    /**
     * Creates a new copyable atomic of the specified type
     */
    copyable_atomic() : _a() {}

    /**
     * Instantiate this by an atomic. Not an atomic operation.
     * @param a the other atomic
     */
    explicit copyable_atomic(const std::atomic<T> &a) : _a(a.load()) {}

    /**
     * Copy constructor. Not an atomic CAS operation.
     * @param other the other copyable atomic
     */
    copyable_atomic(const copyable_atomic &other) : _a(other._a.load()) {}

    /**
     * Copy assign. Not an atomic CAS operation.
     * @param other the other copyable atomic
     * @return this
     */
    copyable_atomic &operator=(const copyable_atomic &other) {
        _a.store(other._a.load());
        return *this;
    }

    /**
     * Const dereference operator, yielding the underlying std atomic.
     * @return the underlying std atomic
     */
    const std::atomic<T> &operator*() const {
        return _a;
    }

    /**
     * Nonconst dereference operator, yielding the underlying std atomic.
     * @return the underlying std atomic
     */
    std::atomic<T> &operator*() {
        return _a;
    }

    /**
     * Const pointer-dereference operator, yielding a pointer to the underlying std atomic.
     * @return a pointer to the underlying std_atomic
     */
    const std::atomic<T> *operator->() const {
        return &_a;
    }

    /**
     * Nonconst pointer-dereference operator, yielding a pointer to the underlying std atomic.
     * @return a pointer to the underlying std_atomic
     */
    std::atomic<T> *operator->() {
        return &_a;
    }
};

}
}
