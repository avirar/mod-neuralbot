#ifndef NEURALBOTSHAREDMEM_H
#define NEURALBOTSHAREDMEM_H

#include "NeuralBotCommon.h"

#include <cstdint>
#include <cstring>
#include <atomic>
#include <string>

constexpr uint32_t SHM_MAGIC   = 0x4E425348; // "NBSH"
constexpr uint32_t SHM_VERSION = 1;
constexpr size_t  SHM_MAX_BOTS = 4096;

// Per-bot observation layout: obs[80] + reward + components[12] = 93 floats
constexpr size_t SHM_OBS_PER_BOT   = OBS_TOTAL_SIZE + 1 + 14;
constexpr size_t SHM_OBS_BYTES     = SHM_OBS_PER_BOT * sizeof(float);
constexpr size_t SHM_ACTIONS_SIZE  = SHM_MAX_BOTS * sizeof(uint8_t);
constexpr size_t SHM_DONES_SIZE    = SHM_MAX_BOTS * sizeof(uint8_t);
constexpr size_t SHM_OBS_REGION_SIZE = SHM_MAX_BOTS * SHM_OBS_BYTES;

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
    uint8_t  _pad[32];       // pad to 64 bytes (8 fields × 4 = 32 + 32 pad = 64)
};

// Shared memory layout (offsets within mmap region):
//   0x0000: NeuralBotSharedControl   (64 bytes)
//   0x0040: _pad_up_to_128           (64 bytes)
//   0x0080: uint8_t  actions[4096]
//   0x1080: _pad_up_to  0x2000       (3968 bytes)
//   0x2000: float    obs_flat[4096 * 93] = 1,524,224 bytes
//   obs_flat_end: uint8_t dones[4096]

constexpr size_t SHM_OFFSET_CONTROL = 0;
constexpr size_t SHM_OFFSET_ACTIONS = 128;
constexpr size_t SHM_OFFSET_OBS     = 0x2000;
constexpr size_t SHM_TOTAL_SIZE     = SHM_OFFSET_OBS + SHM_OBS_REGION_SIZE + SHM_DONES_SIZE;

class NeuralBotSharedMem
{
public:
    static NeuralBotSharedMem& instance();

    bool Create(uint32_t numBots);
    void Destroy();
    bool IsCreated() const { return _ptr != nullptr; }

    // Called from world thread — returns true if Python wrote new actions
    bool TryReadActions(uint8_t* outActions, size_t numBots);

    // Write observations/rewards/dones for all bots, then signal Python
    void WriteObservations(const float* obsFlat, const uint8_t* dones, size_t numBots);
    void SignalObservationsReady();

    // Shutdown
    void SignalShutdown();
    bool IsShutdown() const;

    // Raw pointers
    uint8_t* GetActionPtr() const { return _ptr + SHM_OFFSET_ACTIONS; }
    float*   GetObsPtr()   const { return reinterpret_cast<float*>(_ptr + SHM_OFFSET_OBS); }
    uint8_t* GetDonesPtr() const { return _ptr + SHM_OFFSET_OBS + SHM_OBS_REGION_SIZE; }
    NeuralBotSharedControl* GetControl() const { return reinterpret_cast<NeuralBotSharedControl*>(_ptr); }

private:
    NeuralBotSharedMem() = default;

    uint8_t* _ptr = nullptr;
    std::string _name;
};

#define sNeuralBotShm NeuralBotSharedMem::instance()

#endif
