// Thomas William Unitt-Jones (ACSE-twu18)
/**
 * @file Timer.h
 * @brief Defines the Timer class for measuring elapsed time.
 */

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>
#include <string>

/**
 * @class Timer
 * @brief Provides functionality for measuring elapsed time in a stopwatch-like way.
 *
 * The Timer class allows you to measure elapsed time between `start()` and `stop()` calls.
 * It can be used to profile code segments or measure the execution time of specific parts
 * of the simulation.
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time; ///< Start time point
    std::chrono::duration<double> elapsed; ///< Elapsed time duration
    bool running; ///< Flag indicating whether the timer is running to prevent double run or double stop

public:
    /**
     * @brief Constructor to initialize.
     */
    Timer();

    /**
     * @brief Start the timer.
     */
    void start();

    /**
     * @brief Stop the timer.
     */
    void stop();

    /**
     * @brief Get the elapsed time in seconds.
     * @return Elapsed time in seconds.
     */
    double elapsed_time() const;
};

#endif // TIMER_H
