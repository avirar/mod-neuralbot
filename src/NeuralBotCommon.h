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
#include "NeuralBotFrame.h"
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>

constexpr size_t OBS_PLAYER_STATE_SIZE = 20;
constexpr size_t OBS_NEARBY_UNITS_COUNT = 5;
constexpr size_t OBS_NEARBY_UNIT_FEATURES = 8;
constexpr size_t OBS_COMBAT_STATE_SIZE = 15;
constexpr size_t OBS_QUEST_STATE_SIZE = 10;
constexpr size_t OBS_TOTAL_SIZE = OBS_PLAYER_STATE_SIZE + (OBS_NEARBY_UNITS_COUNT * OBS_NEARBY_UNIT_FEATURES) + OBS_COMBAT_STATE_SIZE + OBS_QUEST_STATE_SIZE;

// ── Action space v2 (DESIGN.md "Action space (v2)") ────────────────────────────
// Point-navigation + entity-index targeting + spellbook-index casting.
// TARGET_ENTITY_i selects the i-th nearest entity exactly as listed in the frame's
// distance-sorted entities[] section; CAST_SPELL_i casts the i-th spellbook entry in
// frame spells[] order. Indices are therefore consistent between observation & action.
constexpr size_t ACTION_COUNT = 41;
constexpr size_t MAX_ENTITY_TARGET_SLOTS = 18; // TARGET_ENTITY_0 .. _17
constexpr size_t MAX_CAST_SLOTS = 8;           // CAST_SPELL_0 .. _7

// Detection radius (yards) for the nearby-units/entities observation and the
// target/attack auto-services. 60 yd left bots unable to see respawned mobs after
// the initial spawn clusters were cleared (kills 0.13/ep -> 0.004/ep), so they
// random-walked away from spawn points. 100 yd lets them sense the next cluster.
constexpr float NB_SENSE_RANGE = 100.0f;

enum NeuralBotAction : uint32
{
    ACTION_NOOP = 0,
    ACTION_MOVE_TO_TARGET = 1,       // MoveChase(selected) — fallback nearest hostile
    ACTION_STOP_MOVE = 2,
    ACTION_MOVE_FORWARD = 3,         // legacy tank control
    ACTION_MOVE_BACKWARD = 4,        // legacy tank control
    ACTION_TURN_LEFT = 5,
    ACTION_TURN_RIGHT = 6,
    ACTION_TARGET_NEAREST_ENEMY = 7,
    ACTION_TARGET_NEAREST_FRIENDLY = 8,
    ACTION_TARGET_NEAREST_CORPSE = 9,
    ACTION_TARGET_ENTITY_0 = 10,     // + i → i-th nearest entity (frame order)
    ACTION_TARGET_ENTITY_LAST = ACTION_TARGET_ENTITY_0 + MAX_ENTITY_TARGET_SLOTS - 1, // 27
    ACTION_ATTACK_START = 28,
    ACTION_ATTACK_STOP = 29,
    ACTION_CAST_SPELL_0 = 30,        // + i → i-th spellbook entry (frame order)
    ACTION_CAST_SPELL_LAST = ACTION_CAST_SPELL_0 + MAX_CAST_SLOTS - 1,                // 37
    ACTION_INTERACT_TARGET = 38,     // context: quest hello/accept, trainer buy, loot
    ACTION_COMPLETE_QUEST = 39,      // turn in at nearest quest ender
    ACTION_LOOT = 40
};

struct NeuralBotObservation
{
    float playerState[OBS_PLAYER_STATE_SIZE] = {0};
    float nearbyUnits[OBS_NEARBY_UNITS_COUNT][OBS_NEARBY_UNIT_FEATURES] = {{0}};
    float combatState[OBS_COMBAT_STATE_SIZE] = {0};
    float questState[OBS_QUEST_STATE_SIZE] = {0};

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
    float enemyProximity = 0.0f;
    float targetAcquired = 0.0f;
    float spellLearned = 0.0f;
    float trainerProximity = 0.0f;
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
