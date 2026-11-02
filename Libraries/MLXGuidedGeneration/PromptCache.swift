// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// The key-value cache of the last prompt a model read on the guided path, which the next
/// guided request reuses up to where the two prompts first differ.
///
/// Requests that share a long start, such as the same instructions, then read only what
/// follows it. The cache always represents a prompt alone: the tokens a request generates are
/// trimmed off once it ends, so the next prompt is compared with a prompt.
///
/// Not thread-safe: use it only inside the model container's serialized access, the same lock
/// a generation holds for its whole run.
package final class PromptCache {
    /// The cache, or `nil` before the first prompt.
    fileprivate(set) var storage: KVCacheStorage?
    /// The prompt tokens the cache represents.
    fileprivate(set) var tokens: [Int] = []

    package init() {}

    /// The number of prompt tokens the cache holds.
    package var count: Int { tokens.count }

    /// Forgets the cached prompt, such as after a failed generation.
    package func clear() {
        storage = nil
        tokens = []
    }

    /// Returns a cache for `prompt`, and how many of its tokens the cache already holds.
    ///
    /// The cached prompt's cache is trimmed to the longest start the two prompts share, keeping
    /// at least the prompt's last token to read so the model returns its logits. A new cache is
    /// made instead when there's nothing to share or the cache can't be rewound exactly: a cache
    /// that compresses or rotates its entries, a different cache plan, or an input with media or
    /// a mask, whose tokens alone don't say what the cache holds.
    func cache(
        for prompt: [Int], input: LMInput, plan: KVCachePlan, makeCache: () throws -> KVCacheStorage
    ) throws -> (storage: KVCacheStorage, reused: Int) {
        guard let storage, storage.plan == plan, storage.processedTokenCount == tokens.count,
            input.image == nil, input.video == nil, input.audio == nil, input.text.mask == nil,
            input.text.tokens.ndim == 1, canTrimPromptCache(storage.cache)
        else {
            clear()
            return (try makeCache(), 0)
        }
        let shared = Swift.min(
            zip(prompt, tokens).prefix { $0 == $1 }.count, prompt.count - 1)
        let trimCount = tokens.count - shared
        guard shared > 0, canTrimPromptCache(storage.cache, numTokens: trimCount),
            storage.trim(trimCount) == trimCount
        else {
            clear()
            return (try makeCache(), 0)
        }
        // The cache now holds only the shared start; it's the request's until it ends.
        clear()
        return (storage, shared)
    }

    /// Keeps `storage` as the cache of `prompt`, trimming off what was generated after it.
    func keep(_ storage: KVCacheStorage, for prompt: [Int]) {
        let generated = storage.processedTokenCount - prompt.count
        guard generated >= 0, canTrimPromptCache(storage.cache, numTokens: generated),
            storage.trim(generated) == generated
        else {
            clear()
            return
        }
        self.storage = storage
        tokens = prompt
    }
}
