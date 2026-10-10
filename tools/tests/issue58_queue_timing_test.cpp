#include "runtime/engine/issue58_queue_timing.h"
#include <cassert>
#include <chrono>
using namespace std::chrono;
int main() {
    using Clock = steady_clock;
    const Clock::time_point origin{};
    assert(ninfer::runtime::issue58_queue_wait_ms<Clock>(origin, origin) == 0);
    assert(ninfer::runtime::issue58_queue_wait_ms<Clock>(origin, origin + milliseconds(1234)) == 1234);
    assert(ninfer::runtime::issue58_queue_wait_ms<Clock>(origin, origin + microseconds(1999)) == 1);
    assert(ninfer::runtime::issue58_queue_wait_ms<Clock>(origin + milliseconds(5), origin) == 0);
}
