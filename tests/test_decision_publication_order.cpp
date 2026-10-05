#include "runtime/engine/decision_commit_publication.h"
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
using namespace ninfer;
using runtime::settle_decision_commit;
int main() {
    DecisionResult result;
    DecisionFieldResult field; field.winner_token = -1; field.winner_index = 1;
    field.candidate_token_paths = {{10, 11}, {42, 43, 44}}; result.fields.push_back(field);
    std::mutex mutex;
    std::condition_variable cv;
    bool in_commit = false, release_commit = false, done = false, failed = false;
    int commits = 0;
    DecisionResult published;
    std::thread worker([&] {
        settle_decision_commit(std::move(result), true,
            [&](std::span<const TokenId> path) {
                if (std::vector<TokenId>(path.begin(), path.end()) != std::vector<TokenId>{42, 43, 44}) throw std::runtime_error("incorrect winner");
                std::unique_lock lock(mutex);
                ++commits;
                in_commit = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_commit; });
            },
            [&](DecisionResult ready) {
                { std::lock_guard lock(mutex); published = std::move(ready); done = true; }
                cv.notify_all();
            },
            [&](std::exception_ptr) {
                { std::lock_guard lock(mutex); failed = true; done = true; }
                cv.notify_all();
            });
    });
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return in_commit; });
        if (done) { release_commit = true; cv.notify_all(); lock.unlock(); worker.join(); throw std::runtime_error("published before commit"); }
        release_commit = true;
        cv.notify_all();
        cv.wait(lock, [&] { return done; });
        if (failed || commits != 1 || published.fields.size() != 1 || published.fields[0].winner_index != 1) {
            lock.unlock(); worker.join(); throw std::runtime_error("published invalid result");
        }
        auto consumer = std::move(published);
        if (consumer.fields.size() != 1) { lock.unlock(); worker.join(); throw std::runtime_error("consumer move lost result"); }
    }
    worker.join();
    int successes = 0, failures = 0;
    DecisionResult second;
    second.fields.push_back(field);
    settle_decision_commit(std::move(second), true,
        [&](std::span<const TokenId>) { throw std::runtime_error("commit failed"); },
        [&](DecisionResult) { ++successes; },
        [&](std::exception_ptr error) {
            try { std::rethrow_exception(error); }
            catch (const std::runtime_error&) { ++failures; }
        });
    if (successes != 0 || failures != 1) throw std::runtime_error("commit error published success");
    std::cout << "decision-publication-order: blocked commit, consumer move, local failure passed\n";
}
