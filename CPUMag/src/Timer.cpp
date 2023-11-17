// Thomas William Unitt-Jones (ACSE-twu18)
#include "Timer.h"

using namespace std;

/**
 * @brief Constructs a Timer object with zero initial elapsed time and stops it.
 */
Timer::Timer() 
    : elapsed(chrono::duration<double>::zero()), running(false) {}

/**
 * @brief Starts the timer if it is not already running.
 *
 * If the timer is already running, a warning is printed to `cerr`.
 */
void Timer::start() {
    if (!running) {
        start_time = chrono::high_resolution_clock::now();
        running = true;
    } else {
        cerr << "Warning: Timer already running." << endl;
    }
}

/**
 * @brief Stops the timer and updates the elapsed time if it was running.
 *
 * If the timer is not running, a warning is printed to `cerr`.
 */
void Timer::stop() {
    if (running) {
        chrono::high_resolution_clock::time_point current_time = chrono::high_resolution_clock::now();
        elapsed += current_time - start_time;
        running = false;
    } else {
        cerr << "Warning: Timer isn't running." << endl;
    }
}

/**
 * @brief Gets the total elapsed time in seconds.
 * @return The total elapsed time in seconds.
 *
 * If the timer is currently running, the elapsed time is updated before returning.
 */
double Timer::elapsed_time() const {
    if (running) {
        chrono::high_resolution_clock::time_point current_time = chrono::high_resolution_clock::now();
        return chrono::duration<double>(elapsed + current_time - start_time).count();
    } else {
        return elapsed.count();
    }
}
