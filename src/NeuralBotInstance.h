#ifndef NEURALBOTINSTANCE_H
#define NEURALBOTINSTANCE_H

#include "NeuralBotCommon.h"
#include "NeuralBotFrame.h"
#include "Player.h"
#include "WorldSession.h"
#include "WorldPacket.h"
#include "QuestDef.h"

#include <mutex>
#include <deque>
#include <set>
#include <map>
#include <array>
#include <atomic>
#include <condition_variable>
#include <string>
#include <chrono>

// Build a structured observation frame for an arbitrary Player (not just a NeuralBot).
// Used by the BC demonstration recorder to observe playerbots; the reward tail is left
// zeroed — progress is reconstructed from self xp/money/level deltas in Python.
void BuildFrameFor(Player* player, NeuralBotFrame& frame, ObjectGuid* entityGuidsOut = nullptr, size_t* entityCountOut = nullptr);

class NeuralBotInstance
{
public:
    NeuralBotInstance(Player* player, WorldSession* session);

    NeuralBotStepResult Step(uint32 action);
    NeuralBotObservation Reset();

    NeuralBotFrameResult StepFrame(uint32 action);
    NeuralBotFrame ResetFrame();

    void RecordOpcode(uint16 opcode);

    void SetSpellSlot(size_t index, uint32 spellId);
    uint32 GetSpellSlot(size_t index) const;
    void GetSpellbook(std::vector<uint32>& spells) const;
    void AutoPopulateSpellSlots();
    void SetSpellSlots(std::vector<uint32> const& spells);

    void OnPlayerJustDied();
    void OnPlayerCreatureKill(Creature* killed = nullptr);
    void OnPlayerLearnSpell(uint32 spellId);
    void ReviveIfDead();
    void StageEpisodeStart();
    void ProcessBotPackets();
    void OnWorldUpdate(uint32 diff);

    void SetAutoQuest(bool enabled) { _autoQuestEnabled = enabled; }
    void AutoAcceptQuests();
    uint32 GetQuestsAutoCompleted() const { return _questAutoCompleted; }

    Player* GetPlayer() const { return _player; }
    WorldSession* GetSession() const { return _session; }
    bool IsReady() const { return _ready; }
    std::string GetName() const;

    void SetMaxSteps(uint32 steps) { _maxSteps = steps; }
    uint32 GetStepCount() const { return _stepCount; }

    // Perf instrumentation: accumulated stage times for the last N StepFrame calls.
    // The manager samples and resets these every PerfLogInterval batches.
    void GetPerfStages(double& actionMs, double& rewardMs, double& buildMs, uint32& steps) const;
    void ClearPerfStages();

    void ResetRewardTracking();

private:
    void BuildObservationInto(NeuralBotObservation& obs);
    void BuildFrame(NeuralBotFrame& frame);
    void ExecuteAction(uint32 action);
    Unit* ResolveFrameEntity(size_t index);
    GameObject* ResolveFrameEntityGO(size_t index);
    uint32 GetFrameSpellId(size_t index);
    void DoInteractWithTarget();
    void ExecuteActionLegacyQuestTurnIn();
    void ExecuteActionLegacyLoot();
    void InjectCMSG(uint16 opcode, std::function<void(WorldPacket&)> filler);
    float ComputeReward(NeuralBotReward& out);
    void AutoCompleteQuests();
    void UpdateIdleTracking(float rewardTotal);
    bool ShouldTerminate(std::string& info);
    void WriteFrameReward(NBStateReward& out, NeuralBotReward const& r);

    Player* _player = nullptr;
    WorldSession* _session = nullptr;
    bool _ready = false;

    uint32 _stepCount = 0;
    uint32 _maxSteps = 3000;

    uint32 _spellSlots[5] = {0};

    std::mutex _opcodeMutex;
    std::deque<uint16> _opcodeHistory;

    float _prevXp = 0.0f;
    float _prevNextLevelXp = 0.0f;
    uint32 _prevLevel = 0;
    float _prevHealth = 0.0f;
    float _killCount = 0.0f;
    bool _diedThisStep = false;

    std::set<uint32> _prevTrackedQuests;
    std::map<uint32, uint8> _prevQuestStatus;
    std::map<uint32, std::array<uint16, 4>> _prevObjectiveCounts;
    float _cachedNearestQGDist = 0.0f;
    float _prevQGDist = 0.0f;
    float _cachedNearestEnemyDist = 0.0f;
    float _prevEnemyDist = 0.0f;
    ObjectGuid _prevTargetGuid;

    bool _autoQuestEnabled = false;
    uint32 _questAutoCompleted = 0;
    uint32 _stepsWithoutReward = 0;

    ObjectGuid _lastKilledGuid;

    // Frame entity cache: guids in exactly the order BuildFrame emitted them
    // (distance-sorted), so TARGET_ENTITY_i resolves to the same entity the policy
    // saw at observation time.
    ObjectGuid _frameEntityGuids[NB_MAX_ENTITIES];
    size_t _frameEntityCount = 0;
    float _prevMoney = 0.0f;
    uint32 _killsThisEpisode = 0;

    float _cachedNearestTrainerDist = 0.0f;
    float _prevTrainerDist = 0.0f;
    ObjectGuid _prevTrainerGuid;
    uint32 _spellsLearnedThisEpisode = 0;

    // Perf instrumentation (accumulated over sampled steps)
    double _accActionMs = 0.0;
    double _accRewardMs = 0.0;
    double _accBuildMs = 0.0;
    uint32 _perfSteps = 0;
};

#endif
