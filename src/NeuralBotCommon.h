#ifndef MOD_NEURALBOT_H
#define MOD_NEURALBOT_H

#include "Common.h"
#include "Player.h"
#include "Unit.h"
#include "Creature.h"
#include "WorldSession.h"
#include "WorldPacket.h"
#include "ObjectGuid.h"
#include "SharedDefines.h"
#include "ScriptMgr.h"
#include "Config.h"
#include "Chat.h"
#include "Map.h"
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>

constexpr size_t OBS_PLAYER_STATE_SIZE = 20;
constexpr size_t OBS_NEARBY_UNITS_COUNT = 10;
constexpr size_t OBS_NEARBY_UNIT_FEATURES = 8;
constexpr size_t OBS_COMBAT_STATE_SIZE = 10;
constexpr size_t OBS_QUEST_STATE_SIZE = 10;
constexpr size_t OBS_OPCODE_HISTORY_SIZE = 64;
constexpr size_t OBS_TOTAL_SIZE = OBS_PLAYER_STATE_SIZE + (OBS_NEARBY_UNITS_COUNT * OBS_NEARBY_UNIT_FEATURES) + OBS_COMBAT_STATE_SIZE + OBS_QUEST_STATE_SIZE + OBS_OPCODE_HISTORY_SIZE;

constexpr size_t ACTION_COUNT = 27;

enum NeuralBotAction : uint32
{
    ACTION_NOOP = 0,
    ACTION_MOVE_FORWARD = 1,
    ACTION_MOVE_BACKWARD = 2,
    ACTION_MOVE_LEFT = 3,
    ACTION_MOVE_RIGHT = 4,
    ACTION_MOVE_FORWARD_LEFT = 5,
    ACTION_MOVE_FORWARD_RIGHT = 6,
    ACTION_MOVE_BACKWARD_LEFT = 7,
    ACTION_MOVE_BACKWARD_RIGHT = 8,
    ACTION_STOP_MOVE = 9,
    ACTION_TURN_LEFT = 10,
    ACTION_TURN_RIGHT = 11,
    ACTION_TARGET_NEAREST_ENEMY = 12,
    ACTION_TARGET_BY_INDEX_0 = 13,
    ACTION_TARGET_BY_INDEX_1 = 14,
    ACTION_TARGET_BY_INDEX_2 = 15,
    ACTION_TARGET_BY_INDEX_3 = 16,
    ACTION_ATTACK_START = 17,
    ACTION_ATTACK_STOP = 18,
    ACTION_CAST_SPELL_1 = 19,
    ACTION_CAST_SPELL_2 = 20,
    ACTION_CAST_SPELL_3 = 21,
    ACTION_INTERACT_LOOT = 22,
    ACTION_STAND_UP = 23,
    ACTION_INTERACT_NPC = 24,
    ACTION_COMPLETE_QUEST = 25,
    ACTION_TARGET_QUEST_GIVER = 26
};

struct NeuralBotObservation
{
    float playerState[OBS_PLAYER_STATE_SIZE] = {0};
    float nearbyUnits[OBS_NEARBY_UNITS_COUNT][OBS_NEARBY_UNIT_FEATURES] = {{0}};
    float combatState[OBS_COMBAT_STATE_SIZE] = {0};
    float questState[OBS_QUEST_STATE_SIZE] = {0};
    int32 opcodeHistory[OBS_OPCODE_HISTORY_SIZE] = {0};

    void ToFloatArray(float* out) const;
};

struct NeuralBotReward
{
    float xpDelta = 0.0f;
    float damageDealt = 0.0f;
    float damageTaken = 0.0f;
    float killReward = 0.0f;
    float deathPenalty = 0.0f;
    float lootReward = 0.0f;
    float questAccepted = 0.0f;
    float questCompleted = 0.0f;
    float questProximity = 0.0f;
    float questProgress = 0.0f;
    float timePenalty = -0.001f;
    float total = 0.0f;
};

struct NeuralBotStepResult
{
    NeuralBotObservation observation;
    NeuralBotReward reward;
    bool done = false;
    std::string info;
};

#endif
