#ifndef NEURALBOTSHAREDMEM_H
#define NEURALBOTSHAREDMEM_H

#include "NeuralBotCommon.h"

#include <cstdint>
#include <cstring>
#include <atomic>
#include <string>

constexpr uint32_t SHM_MAGIC   = 0x4E425348; // "NBSH"
constexpr uint32_t SHM_VERSION = 2;
constexpr size_t  SHM_MAX_BOTS = 4096;

// Per-bot observation: one packed NeuralBotFrame (structured entity-centric state).
constexpr size_t SHM_FRAME_BYTES      = sizeof(NeuralBotFrame);
constexpr size_t SHM_ACTIONS_SIZE     = SHM_MAX_BOTS * sizeof(uint8_t);
constexpr size_t SHM_DONES_SIZE       = SHM_MAX_BOTS * sizeof(uint8_t);
constexpr size_t SHM_FRAME_REGION_SIZE = SHM_MAX_BOTS * SHM_FRAME_BYTES;

// Directional flags — one writer per field, no read-modify-write races
struct NeuralBotSharedControl
{
    uint32_t magic;          // always SHM_MAGIC
    uint32_t version;        // SHM_VERSION
    uint32_t num_bots;       // actual bot count (<= SHM_MAX_BOTS)
    uint32_t step_count;     // monotonic, incremented each round by C++
    int32_t  eventfd;        // eventfd fd for Python wake-up, -1 if none
    uint32_t actions_ready;  // Python→C++: Python writes 1, C++ clears to 0
    uint32_t obs_ready;      // C++→Python: C++ writes 1, Python reads
    uint32_t shutdown;       // either side sets to 1 to request exit
    uint32_t frame_bytes;    // sizeof(NeuralBotFrame) — schema negotiation
    uint8_t  _pad[28];       // pad to 64 bytes
};

// Shared memory layout (offsets within mmap region):
//   0x0000: NeuralBotSharedControl   (64 bytes)
//   0x0040: _pad_up_to_128           (64 bytes)
//   0x0080: uint8_t  actions[4096]
//   0x1080: _pad_up_to  0x2000       (3968 bytes)
//   0x2000: NeuralBotFrame frames[4096] (packed, frame_bytes each)
//   frames_end: uint8_t dones[4096]

constexpr size_t SHM_OFFSET_CONTROL = 0;
constexpr size_t SHM_OFFSET_ACTIONS = 128;
constexpr size_t SHM_OFFSET_OBS     = 0x2000;
constexpr size_t SHM_TOTAL_SIZE     = SHM_OFFSET_OBS + SHM_FRAME_REGION_SIZE + SHM_DONES_SIZE;

class NeuralBotSharedMem
{
public:
    static NeuralBotSharedMem& instance();

    bool Create(uint32_t numBots);
    void Destroy();
    bool IsCreated() const { return _ptr != nullptr; }

    // Called from world thread — returns true if Python wrote new actions
    bool TryReadActions(uint8_t* outActions, size_t numBots);

    // Pipelined protocol: true while Python has not yet consumed the last frame batch.
    // C++ must not overwrite frames (or dones) while this is set — the Python reader
    // thread clears it within ~1ms of harvesting.
    bool ObservationsPending() const;

    // Write observation frames/rewards/dones for all bots, then signal Python
    void WriteFrames(const uint8_t* frames, const uint8_t* dones, size_t numBots);
    void SignalObservationsReady();

    // Shutdown
    void SignalShutdown();
    bool IsShutdown() const;

    // Raw pointers
    uint8_t* GetActionPtr() const { return _ptr + SHM_OFFSET_ACTIONS; }
    uint8_t* GetFramesPtr() const { return _ptr + SHM_OFFSET_OBS; }
    uint8_t* GetDonesPtr() const { return _ptr + SHM_OFFSET_OBS + SHM_FRAME_REGION_SIZE; }
    NeuralBotSharedControl* GetControl() const { return reinterpret_cast<NeuralBotSharedControl*>(_ptr); }

private:
    NeuralBotSharedMem() = default;

    uint8_t* _ptr = nullptr;
    std::string _name;
};

#define sNeuralBotShm NeuralBotSharedMem::instance()

#endif
