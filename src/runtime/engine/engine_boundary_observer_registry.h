#pragma once

// Private test hook. Register before Engine construction; no public Engine API.
#include <cstdint>

#include <functional>
#include <memory>

namespace ninfer {
namespace runtime { struct BoundaryObservation; }
class BoundaryObserverRegistry {
public:
    using Callback = std::function<void(const runtime::BoundaryObservation&)>;
    using Token = std::uint64_t;

    // One active registration per process; returns a non-reused identity token.
    static Token register_observer(Callback callback);
    // Copies shared ownership under the registry lock, never invokes under that lock.
    static std::shared_ptr<const Callback> snapshot();
    // Rejects stale/mismatched tokens, including after replacement registration.
    static void unregister(Token token);
};
} // namespace ninfer
