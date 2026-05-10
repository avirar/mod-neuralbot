#ifndef NEURALBOTMGR_H
#define NEURALBOTMGR_H

#include "NeuralBotCommon.h"
#include "Player.h"
#include "WorldSession.h"
#include "WorldPacket.h"

#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <set>
#include <map>
#include <array>

class NeuralBotMgr
{
public:
    static NeuralBotMgr& instance();

    void Initialize();
    void Shutdown();

    bool IsEnabled() const { return _enabled; }

    void OnWorldUpdate(uint32 diff);
    void OnPlayerbotPacketSent(Player* player, WorldPacket const* packet);
    void OnPlayerLogin(Player* player);
    void OnPlayerJustDied(Player* player);
    void OnPlayerCreatureKill(Player* killer, Creature* killed);
    void OnPlayerAfterUpdate(Player* player, uint32 diff);

    void HandleBotLogin();
    void ScheduleLogin();
    void DoLogin();

    Player* GetBotPlayer();
    WorldSession* GetBotSession();

    NeuralBotStepResult Step(uint32 action);
    NeuralBotObservation Reset();

    void RecordOpcode(uint16 opcode);

    void SetSpellSlot(size_t index, uint32 spellId);
    uint32 GetSpellSlot(size_t index) const;
    void GetSpellbook(std::vector<uint32>& spells);
    void AutoPopulateSpellSlots();
    void SetSpellSlots(std::vector<uint32> const& spells);

private:
    NeuralBotMgr() = default;

    void BuildObservationInto(NeuralBotObservation& obs);
    void ExecuteAction(uint32 action);
    void InjectCMSG(uint16 opcode, std::function<void(WorldPacket&)> filler);
    void ProcessBotPackets();

    float ComputeReward();
    void ResetRewardTracking();

    bool _enabled = false;
    bool _botReady = false;
    bool _loginScheduled = false;
    uint32 _botAccountId = 0;
    std::string _botCharacterName;
    ObjectGuid _botGuid;
    Player* _botPlayer = nullptr;
    WorldSession* _botSession = nullptr;
    ObjectGuid::LowType _botGuidLow = 0;

    uint32 _stepCount = 0;
    uint32 _maxSteps = 1000;
    uint32 _tickRateMs = 50;
    uint32 _loginTimer = 0;

    uint32 _spellSlots[5] = {0};

    std::mutex _opcodeMutex;
    std::deque<uint16> _opcodeHistory;

    float _prevXp = 0.0f;
    float _prevHealth = 0.0f;
    float _killCount = 0.0f;
    bool _diedThisStep = false;

    // Quest state tracking
    std::set<uint32> _prevTrackedQuests;
    std::map<uint32, uint8> _prevQuestStatus;
    std::map<uint32, std::array<uint16, 4>> _prevObjectiveCounts;
    float _cachedNearestQGDist = 0.0f;

    std::mutex _stepMutex;
    std::condition_variable _stepCv;
    std::atomic<bool> _stepPending{false};
    uint32 _pendingAction = ACTION_NOOP;
    NeuralBotStepResult _lastResult;
};

#define sNeuralBotMgr NeuralBotMgr::instance()

#endif
