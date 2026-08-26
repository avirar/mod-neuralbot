#ifndef NEURALBOTMGR_H
#define NEURALBOTMGR_H

#include "NeuralBotCommon.h"
#include "NeuralBotInstance.h"
#include "Player.h"
#include "WorldSession.h"
#include "WorldPacket.h"

#include <mutex>
#include <deque>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <set>
#include <map>
#include <array>
#include <string>
#include <future>
#include <cstdio>
#include <chrono>

class NeuralBotMgr
{
public:
    static NeuralBotMgr& instance();

    void Initialize();
    void Shutdown();

    bool IsEnabled() const { return _enabled; }

    void OnWorldUpdate(uint32 diff);
    void OnPlayerbotPacketSent(Player* player, WorldPacket const* packet);
    void OnPlayerbotActionExecuted(Player* player, std::string const& actionName, ObjectGuid target);
    void OnPlayerLogin(Player* player);
    void OnPlayerJustDied(Player* player);
    void OnPlayerCreatureKill(Player* killer, Creature* killed);
    void OnPlayerLearnSpell(Player* player, uint32 spellId);
    void OnPlayerAfterUpdate(Player* player, uint32 diff);

    NeuralBotStepResult Step(std::string const& botName, uint32 action);
    NeuralBotObservation Reset(std::string const& botName);
    void RecordOpcodeFor(Player* player, uint16 opcode);

    NeuralBotInstance* GetInstance(std::string const& botName);
    std::vector<std::string> GetBotNames() const;
    size_t GetBotCount() const { return _instances.size(); }
    void SpawnAndLoginBots();

private:
    NeuralBotMgr() = default;
    NeuralBotMgr(NeuralBotMgr const&) = delete;
    NeuralBotMgr& operator=(NeuralBotMgr const&) = delete;

    void LoginAllBots();
    bool ProcessPendingRequests();
    void ProcessSharedMemoryStep();

    void InitBcRecorder();
    void CloseBcRecorder();

    bool _enabled = false;
    bool _autoQuest = false;
    bool _curriculumStaging = false;
    bool _levelBandRespawn = false;
    bool _cleanupOnStartup = false;

    // Behavior-cloning demonstration recorder (playerbots): appends fixed-size
    // NeuralBotBcRecord entries to the file named by NeuralBot.BcRecordPath.
    FILE* _bcFile = nullptr;
    uint32 _bcRecordEvery = 1;
    uint64 _bcSeq = 0;
    uint64 _bcCounter = 0;

    std::map<std::string, NeuralBotInstance*> _instances;
    std::map<ObjectGuid, NeuralBotInstance*> _instancesByGuid;

    // Shared memory: ordered bot access
    std::vector<std::string> _botOrder;
    uint32_t _botCount = 0;

    // Perf instrumentation (batched SHM step cycle): accumulated over N batches,
    // dumped as module.neuralbot.perf every NeuralBot.PerfLogInterval batches.
    std::chrono::steady_clock::time_point _lastBatchDone;
    double _accCycleMs = 0.0;
    double _accStepMs = 0.0;
    uint32 _perfSamples = 0;
    uint32 _perfLogInterval = 250;

    struct PendingLogin
    {
        uint32 accountId;
        ObjectGuid guid;
        std::string name;
        WorldSession* session;
    };
    std::vector<PendingLogin> _pendingLogins;

    struct PendingStep
    {
        std::string botName;
        uint32 action;
        std::promise<NeuralBotStepResult> promise;
    };
    struct PendingReset
    {
        std::string botName;
        std::promise<NeuralBotObservation> promise;
    };
    std::mutex _queueMutex;
    std::vector<PendingStep> _pendingSteps;
    std::vector<PendingReset> _pendingResets;
};

#define sNeuralBotMgr NeuralBotMgr::instance()

#endif
