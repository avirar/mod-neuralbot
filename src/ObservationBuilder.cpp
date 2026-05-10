#include "NeuralBotCommon.h"
#include "Player.h"
#include "Unit.h"
#include "Creature.h"
#include "Cell.h"
#include "CellImpl.h"
#include "GridNotifiers.h"
#include "GridNotifiersImpl.h"
#include "Map.h"

#include <cmath>
#include <algorithm>

void NeuralBotObservation::ToFloatArray(float* out) const
{
    size_t offset = 0;
    std::memcpy(out + offset, playerState, OBS_PLAYER_STATE_SIZE * sizeof(float));
    offset += OBS_PLAYER_STATE_SIZE;
    std::memcpy(out + offset, nearbyUnits, OBS_NEARBY_UNITS_COUNT * OBS_NEARBY_UNIT_FEATURES * sizeof(float));
    offset += OBS_NEARBY_UNITS_COUNT * OBS_NEARBY_UNIT_FEATURES;
    std::memcpy(out + offset, combatState, OBS_COMBAT_STATE_SIZE * sizeof(float));
    offset += OBS_COMBAT_STATE_SIZE;
    std::memcpy(out + offset, questState, OBS_QUEST_STATE_SIZE * sizeof(float));
}
