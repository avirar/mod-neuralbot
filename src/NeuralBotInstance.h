#ifndef NEURALBOTINSTANCE_H
#define NEURALBOTINSTANCE_H

#include "NeuralBotCommon.h"
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

class NeuralBotInstance
{
public:
    NeuralBotInstance(Player* player, WorldSession* session);

    NeuralBotStepResult Step(uint32 action);
    NeuralBotObservation Reset();
    void RecordOpcode(uint16 opcode);

    void SetSpellSlot(size_t index, uint32 spellId);
    uint32 GetSpellSlot(size_t index) const;
    void GetSpellbook(std::vector<uint32>& spells) const;
    void AutoPopulateSpellSlots();
    void SetSpellSlots(std::vector<uint32> const& spells);

    void OnPlayerJustDied();
    void OnPlayerCreatureKill();
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

private:
    void BuildObservationInto(NeuralBotObservation& obs);
    void ExecuteAction(uint32 action);
    void InjectCMSG(uint16 opcode, std::function<void(WorldPacket&)> filler);
    float ComputeReward(NeuralBotReward& out);
    void ResetRewardTracking();
    void AutoCompleteQuests();

    Player* _player = nullptr;
    WorldSession* _session = nullptr;
    bool _ready = false;

    uint32 _stepCount = 0;
    uint32 _maxSteps = 1000;

    uint32 _spellSlots[5] = {0};

    std::mutex _opcodeMutex;
    std::deque<uint16> _opcodeHistory;

    float _prevXp = 0.0f;
    float _prevHealth = 0.0f;
    float _killCount = 0.0f;
    bool _diedThisStep = false;

    std::set<uint32> _prevTrackedQuests;
    std::map<uint32, uint8> _prevQuestStatus;
    std::map<uint32, std::array<uint16, 4>> _prevObjectiveCounts;
    float _cachedNearestQGDist = 0.0f;
    float _prevQGDist = 0.0f;

    bool _autoQuestEnabled = false;
    uint32 _questAutoCompleted = 0;
    uint32 _stepsWithoutReward = 0;
};

#endif
