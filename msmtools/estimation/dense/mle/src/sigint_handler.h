/* * This file is part of MSMTools.
 *
 * Copyright (c) 2015, 2014 Computational Molecular Biology Group
 *
 * MSMTools is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SIGINT_HANDLER_H
#define SIGINT_HANDLER_H

#include <signal.h>

static volatile sig_atomic_t interrupted;
static void (*old_handler)(int);

static void signal_handler(int signo) {
  interrupted = 1;
}

static void sigint_on(void) {
  interrupted = 0;
  old_handler = signal(SIGINT, signal_handler);
}

static void sigint_off(void) {
  if(old_handler != SIG_ERR) {
    signal(SIGINT, old_handler);
    if(interrupted) raise(SIGINT);
  }
}

#endif
