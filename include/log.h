/** @file log.h
 *  @brief Macros for different printing types.
 *
 *  This contains the definition of different  
 *  printing types.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

#ifndef _LOG_H
#define _LOG_H

/* -- Includes -- */

/* libc includes. */
#include <stdio.h>

/* -- Defines -- */

/* ANSI color defines. */
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

/** @brief Displays cosmetic log.
 *
 *  @param msg The message to display.
 *  @param ... The other eventual arguments.
 */
#define COS(msg, ...) printf(ANSI_COLOR_BLUE msg ANSI_COLOR_RESET "\n",##__VA_ARGS__)

/** @brief Displays informative log.
 *
 *  @param msg The message to display.
 *  @param ... The other eventual arguments.
 */
#define INF(msg, ...) printf(ANSI_COLOR_GREEN msg ANSI_COLOR_RESET "\n",##__VA_ARGS__)

/** @brief Displays classic log.
 *
 *  @param msg The message to display.
 *  @param ... The other eventual arguments.
 */
#define SAY(msg, ...) printf(msg"\n",##__VA_ARGS__)

/** @brief Displays warning log.
 *
 *  @param msg The message to display.
 *  @param ... The other eventual arguments.
 */
#define WRN(msg, ...) printf(ANSI_COLOR_YELLOW msg ANSI_COLOR_RESET "\n",##__VA_ARGS__)

/** @brief Displays error log.
 *
 *  @param msg The message to display.
 *  @param ... The other eventual arguments.
 */
#define ERR(msg, ...) printf(ANSI_COLOR_RED msg ANSI_COLOR_RESET "\n",##__VA_ARGS__)


#endif /* _LOG_H */
