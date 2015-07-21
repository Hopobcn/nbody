#ifdef WIN32
#pragma warning(disable:4996)
#endif

// includes, project
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <exception.hpp>
#include <math.h>

#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>

// includes, timer, string parsing, image helpers
#include <helper_timer.hpp>   // helper functions for timers
#include <helper_string.hpp>  // helper functions for string parsing
#include <helper_image.hpp>   // helper functions for image compare, dump, data comparisons

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif
