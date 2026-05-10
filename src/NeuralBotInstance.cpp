#include "NeuralBotInstance.h"
#include "QuestDef.h"
#include "ObjectMgr.h"
#include "GridNotifiers.h"
#include "GridNotifiersImpl.h"
#include "Cell.h"
#include "CellImpl.h"
#include "MotionMaster.h"
#include "SpellMgr.h"
#include "Log.h"
#include "Opcodes.h"
#include "Map.h"

#include <algorithm>
#include <cmath>

NeuralBotInstance::NeuralBotInstance(Player* player, WorldSession* session)
    : _player(player), _session(session), _ready(player && session)
{
    if (_ready)
    {
        AutoPopulateSpellSlots();
        ResetRewardTracking();
        LOG_INFO("module.neuralbot", "Instance created for '{}' GUID: {}", GetName(), _player->GetGUID().GetCounter());
    }
}

std::string NeuralBotInstance::GetName() const
{
    return _player ? _player->GetName() : "?";
}

void NeuralBotInstance::SetSpellSlot(size_t index, uint32 spellId)
{
    if (index < 5)
        _spellSlots[index] = spellId;
}

uint32 NeuralBotInstance::GetSpellSlot(size_t index) const
{
    return index < 5 ? _spellSlots[index] : 0;
}

void NeuralBotInstance::GetSpellbook(std::vector<uint32>& spells) const
{
    if (!_player)
        return;
    spells.clear();
    PlayerSpellMap const& map = _player->GetSpellMap();
    for (auto const& [spellId, state] : map)
    {
        if (state->State == PLAYERSPELL_REMOVED)
            continue;
        spells.push_back(spellId);
    }
}

void NeuralBotInstance::SetSpellSlots(std::vector<uint32> const& spells)
{
    std::fill(std::begin(_spellSlots), std::end(_spellSlots), 0u);
    for (size_t i = 0; i < 5 && i < spells.size(); ++i)
        _spellSlots[i] = spells[i];
}

void NeuralBotInstance::AutoPopulateSpellSlots()
{
    if (!_player)
        return;

    PlayerSpellMap const& map = _player->GetSpellMap();
    std::vector<uint32> combatSpells;

    for (auto const& [spellId, state] : map)
    {
        if (state->State == PLAYERSPELL_REMOVED)
            continue;
        SpellInfo const* info = sSpellMgr->GetSpellInfo(spellId);
        if (!info || info->IsPassive() || info->HasAttribute(SPELL_ATTR0_DO_NOT_DISPLAY))
            continue;
        if (info->HasEffect(SPELL_EFFECT_APPLY_AURA))
            continue;
        if (!info->HasEffect(SPELL_EFFECT_WEAPON_DAMAGE) &&
            !info->HasEffect(SPELL_EFFECT_SCHOOL_DAMAGE) &&
            !info->HasEffect(SPELL_EFFECT_ATTACK) &&
            info->SpellFamilyName == 0)
            continue;
        if (_player->HasSpellCooldown(spellId))
            continue;
        combatSpells.push_back(spellId);
    }

    std::fill(std::begin(_spellSlots), std::end(_spellSlots), 0u);
    for (size_t i = 0; i < 5 && i < combatSpells.size(); ++i)
        _spellSlots[i] = combatSpells[i];
}

void NeuralBotInstance::RecordOpcode(uint16 opcode)
{
    std::lock_guard<std::mutex> lock(_opcodeMutex);
    _opcodeHistory.push_back(opcode);
    if (_opcodeHistory.size() > 64)
        _opcodeHistory.pop_front();
}

void NeuralBotInstance::InjectCMSG(uint16 opcode, std::function<void(WorldPacket&)> filler)
{
    if (!_session)
        return;

    WorldPacket* pkt = new WorldPacket(opcode, 64);
    if (filler)
        filler(*pkt);
    pkt->resize(pkt->wpos());
    _session->QueuePacket(pkt);
}

void NeuralBotInstance::ProcessBotPackets()
{
    if (!_session)
        return;

    WorldPacket* packet;
    auto& queue = _session->GetPacketQueue();
    while (queue.next(packet))
    {
        if (!packet)
            continue;
        try
        {
            auto opcode = static_cast<OpcodeClient>(packet->GetOpcode());
            auto const* opHandle = opcodeTable[opcode];
            if (opHandle)
                opHandle->Call(_session, *packet);
        }
        catch (ByteBufferException const& e)
        {
            LOG_ERROR("module.neuralbot", "ByteBufferException for opcode {}: {}", packet->GetOpcode(), e.what());
        }
        delete packet;
    }
}

void NeuralBotInstance::OnPlayerJustDied()
{
    _diedThisStep = true;
}

void NeuralBotInstance::OnPlayerCreatureKill()
{
    _killCount += 1.0f;
}

void NeuralBotInstance::BuildObservationInto(NeuralBotObservation& obs)
{
    Player* bot = _player;
    if (!bot)
        return;

    float maxHp = static_cast<float>(bot->GetMaxHealth());
    obs.playerState[0] = maxHp > 0 ? static_cast<float>(bot->GetHealth()) / maxHp : 0.0f;

    float maxMp = static_cast<float>(bot->GetMaxPower(POWER_MANA));
    obs.playerState[1] = maxMp > 0 ? static_cast<float>(bot->GetPower(POWER_MANA)) / maxMp : 0.0f;

    float maxRage = static_cast<float>(bot->GetMaxPower(POWER_RAGE));
    obs.playerState[2] = maxRage > 0 ? static_cast<float>(bot->GetPower(POWER_RAGE)) / 1000.0f : 0.0f;

    obs.playerState[3] = static_cast<float>(bot->GetLevel()) / 80.0f;

    Position pos = bot->GetPosition();
    obs.playerState[4] = pos.GetPositionX() / 10000.0f;
    obs.playerState[5] = pos.GetPositionY() / 10000.0f;
    obs.playerState[6] = pos.GetPositionZ() / 1000.0f;
    obs.playerState[7] = pos.GetOrientation() / (2.0f * static_cast<float>(M_PI));

    obs.playerState[8] = static_cast<float>(bot->GetMoney()) / 1000000.0f;
    obs.playerState[9] = static_cast<float>(bot->GetZoneId()) / 5000.0f;

    Unit* target = bot->GetSelectedUnit();
    obs.playerState[10] = target ? 1.0f : 0.0f;
    if (target)
    {
        float tMaxHp = static_cast<float>(target->GetMaxHealth());
        obs.playerState[11] = tMaxHp > 0 ? static_cast<float>(target->GetHealth()) / tMaxHp : 0.0f;
        obs.playerState[12] = static_cast<float>(target->GetLevel()) / 80.0f;
        obs.playerState[13] = std::min(bot->GetDistance(target) / 100.0f, 1.0f);
        obs.playerState[14] = target->IsHostileTo(bot) ? 1.0f : 0.0f;
    }

    obs.playerState[15] = bot->IsAlive() ? 1.0f : 0.0f;
    obs.playerState[16] = bot->isMoving() ? 1.0f : 0.0f;

    uint32 xp = bot->GetUInt32Value(PLAYER_XP);
    obs.playerState[17] = static_cast<float>(xp) / 100000.0f;

    uint32 pxp = bot->GetUInt32Value(PLAYER_NEXT_LEVEL_XP);
    obs.playerState[18] = pxp > 0 ? static_cast<float>(xp) / static_cast<float>(pxp) : 0.0f;

    obs.playerState[19] = bot->GetMapId() / 1000.0f;

    // --- Nearby units ---
    std::vector<std::pair<Unit*, float>> nearbyUnits;
    if (bot->IsInWorld())
    {
        float range = 40.0f;
        std::list<Unit*> targets;
        Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, range);
        Acore::UnitListSearcher<decltype(check)> searcher(bot, targets, check);
        Cell::VisitObjects(bot, searcher, range);

        for (Unit* u : targets)
            if (u && u->IsInWorld())
                nearbyUnits.push_back({u, bot->GetDistance(u)});

        std::sort(nearbyUnits.begin(), nearbyUnits.end(),
            [](auto const& a, auto const& b) { return a.second < b.second; });
    }

    for (size_t i = 0; i < OBS_NEARBY_UNITS_COUNT; ++i)
    {
        if (i < nearbyUnits.size())
        {
            Unit* u = nearbyUnits[i].first;
            float uMaxHp = static_cast<float>(u->GetMaxHealth());
            obs.nearbyUnits[i][0] = uMaxHp > 0 ? static_cast<float>(u->GetHealth()) / uMaxHp : 0.0f;
            obs.nearbyUnits[i][1] = static_cast<float>(u->GetLevel()) / 80.0f;
            obs.nearbyUnits[i][2] = std::min(nearbyUnits[i].second / 40.0f, 1.0f);
            obs.nearbyUnits[i][3] = u->IsHostileTo(bot) ? 1.0f : 0.0f;
            obs.nearbyUnits[i][4] = u->IsInCombat() ? 1.0f : 0.0f;
            obs.nearbyUnits[i][5] = u->IsPlayer() ? 1.0f : 0.0f;
            obs.nearbyUnits[i][6] = static_cast<float>(u->GetEntry()) / 50000.0f;
            obs.nearbyUnits[i][7] = u->IsAlive() ? 1.0f : 0.0f;
        }
    }

    // --- Combat state ---
    obs.combatState[0] = bot->IsInCombat() ? 1.0f : 0.0f;
    obs.combatState[1] = bot->HasUnitState(UNIT_STATE_CASTING) ? 1.0f : 0.0f;
    obs.combatState[2] = bot->HasAuraType(SPELL_AURA_MOD_STUN) ? 1.0f : 0.0f;
    obs.combatState[3] = bot->isDead() ? 1.0f : 0.0f;
    obs.combatState[4] = bot->IsStandState() ? 1.0f : 0.0f;

    for (size_t i = 0; i < 5; ++i)
    {
        uint32 spellId = _spellSlots[i];
        obs.combatState[5 + i] = (spellId > 0 && !bot->HasSpellCooldown(spellId)) ? 1.0f : 0.0f;
    }

    // --- Quest state ---
    {
        uint32 questCount = 0;
        float bestProgress = 0.0f;
        std::array<float, 4> topProgress = {0.0f, 0.0f, 0.0f, 0.0f};
        bool hasCompletable = false;
        float nearestQGDist = 0.0f;
        float nearestQEDist = 0.0f;

        for (uint32 slot = 0; slot < 25; ++slot)
        {
            uint32 qid = bot->GetQuestSlotQuestId(slot);
            if (!qid)
                continue;
            questCount++;
            if (bot->GetQuestStatus(qid) == QUEST_STATUS_COMPLETE)
                hasCompletable = true;

            Quest const* quest = sObjectMgr->GetQuestTemplate(qid);
            if (quest)
            {
                uint32 curSum = 0, reqSum = 0;
                for (uint8 i = 0; i < 4; ++i)
                {
                    curSum += bot->GetQuestSlotCounter(slot, i);
                    reqSum += quest->RequiredNpcOrGo[i];
                }
                float progress = reqSum > 0 ? std::min(static_cast<float>(curSum) / static_cast<float>(reqSum), 1.0f) : 0.0f;
                if (progress > bestProgress)
                    bestProgress = progress;
                for (auto& p : topProgress)
                    if (progress > p)
                        std::swap(progress, p);
            }
        }

        if (bot->IsInWorld())
        {
            float range = 40.0f;
            std::list<Unit*> friendlies;
            Acore::AnyFriendlyUnitInObjectRangeCheck friendlyCheck(bot, bot, range);
            Acore::UnitListSearcher<decltype(friendlyCheck)> searcher(bot, friendlies, friendlyCheck);
            Cell::VisitObjects(bot, searcher, range);

            for (Unit* u : friendlies)
            {
                Creature* c = u->ToCreature();
                if (!c) continue;
                uint32 entry = c->GetEntry();
                float dist = bot->GetDistance(c);

                auto startBounds = sObjectMgr->GetCreatureQuestRelationBounds(entry);
                if (startBounds.first != startBounds.second)
                    if (nearestQGDist == 0.0f || dist < nearestQGDist)
                        nearestQGDist = dist;

                auto endBounds = sObjectMgr->GetCreatureQuestInvolvedRelationBounds(entry);
                if (endBounds.first != endBounds.second)
                    if (nearestQEDist == 0.0f || dist < nearestQEDist)
                        nearestQEDist = dist;
            }
        }

        _cachedNearestQGDist = nearestQGDist;

        obs.questState[0] = static_cast<float>(questCount) / 25.0f;
        obs.questState[1] = nearestQGDist > 0.0f ? std::min(nearestQGDist / 40.0f, 1.0f) : 1.0f;
        obs.questState[2] = nearestQGDist > 0.0f && nearestQGDist <= 40.0f ? 1.0f : 0.0f;
        obs.questState[3] = nearestQEDist > 0.0f ? std::min(nearestQEDist / 40.0f, 1.0f) : 1.0f;
        obs.questState[4] = hasCompletable ? 1.0f : 0.0f;
        obs.questState[5] = bestProgress;
        obs.questState[6] = topProgress[0];
        obs.questState[7] = topProgress[1];
        obs.questState[8] = topProgress[2];
        obs.questState[9] = topProgress[3];
    }
}

float NeuralBotInstance::ComputeReward(NeuralBotReward& out)
{
    out = NeuralBotReward{};
    Player* bot = _player;
    if (!bot) return 0.0f;

    float reward = 0.0f;

    float curXp = static_cast<float>(bot->GetUInt32Value(PLAYER_XP));
    float xpReward = (curXp - _prevXp) / 100.0f;
    reward += xpReward;
    out.xpDelta = xpReward;
    _prevXp = curXp;

    float killReward = _killCount * 5.0f;
    reward += killReward;
    out.killReward = killReward;
    _killCount = 0.0f;

    float curHp = static_cast<float>(bot->GetHealth());
    float hpDelta = _prevHealth - curHp;
    float damagePenalty = 0.0f;
    if (hpDelta > 0)
    {
        damagePenalty = hpDelta / static_cast<float>(bot->GetMaxHealth()) * 0.5f;
        reward -= damagePenalty;
    }
    out.damageTaken = damagePenalty;
    _prevHealth = curHp;

    if (_diedThisStep)
    {
        out.deathPenalty = 10.0f;
        reward -= 10.0f;
        _diedThisStep = false;
    }

    float questAcceptedReward = 0.0f, questCompletedReward = 0.0f;
    float questProgressReward = 0.0f, questProximityReward = 0.0f;

    {
        std::set<uint32> curActive;
        std::map<uint32, uint8> curStatus;
        std::map<uint32, std::array<uint16, 4>> curCounters;

        for (uint32 slot = 0; slot < 25; ++slot)
        {
            uint32 qid = bot->GetQuestSlotQuestId(slot);
            if (!qid) continue;
            curActive.insert(qid);
            curStatus[qid] = static_cast<uint8>(bot->GetQuestStatus(qid));
            curCounters[qid] = {{
                bot->GetQuestSlotCounter(slot, 0),
                bot->GetQuestSlotCounter(slot, 1),
                bot->GetQuestSlotCounter(slot, 2),
                bot->GetQuestSlotCounter(slot, 3)
            }};
        }

        for (uint32 qid : _prevTrackedQuests)
            if (curActive.find(qid) == curActive.end())
                curStatus[qid] = static_cast<uint8>(bot->GetQuestStatus(qid));

        for (uint32 qid : curActive)
            if (_prevTrackedQuests.find(qid) == _prevTrackedQuests.end())
            {
                reward += 5.0f;
                questAcceptedReward += 5.0f;
            }

        for (auto const& [qid, prevStatus] : _prevQuestStatus)
        {
            auto it = curStatus.find(qid);
            if (it != curStatus.end())
            {
                uint8 cur = it->second;
                if (prevStatus == static_cast<uint8>(QUEST_STATUS_COMPLETE) &&
                    cur == static_cast<uint8>(QUEST_STATUS_REWARDED))
                {
                    reward += 20.0f;
                    questCompletedReward += 20.0f;
                }
            }
        }

        for (auto const& [qid, prevCounts] : _prevObjectiveCounts)
        {
            auto it = curCounters.find(qid);
            if (it != curCounters.end())
                for (int i = 0; i < 4; ++i)
                    if (it->second[i] > prevCounts[i])
                    {
                        float tick = 0.5f * static_cast<float>(it->second[i] - prevCounts[i]);
                        reward += tick;
                        questProgressReward += tick;
                    }
        }

        if (_cachedNearestQGDist > 0.0f)
        {
            if (_cachedNearestQGDist < 5.0f)
                questProximityReward = 0.5f * (1.0f - _cachedNearestQGDist / 5.0f);
            else if (_cachedNearestQGDist < 10.0f)
                questProximityReward = 0.25f;
            else if (_cachedNearestQGDist < 20.0f)
                questProximityReward = 0.1f;
            else
                questProximityReward = 0.05f;

            if (_prevQGDist > 0.0f && _cachedNearestQGDist < _prevQGDist)
            {
                float pct = (_prevQGDist - _cachedNearestQGDist) / _prevQGDist;
                questProximityReward += std::min(pct * 0.05f, 0.1f);
            }
            reward += questProximityReward;
            _prevQGDist = _cachedNearestQGDist;
        }

        _prevTrackedQuests.insert(curActive.begin(), curActive.end());
        _prevQuestStatus = std::move(curStatus);
        _prevObjectiveCounts = std::move(curCounters);
    }

    out.questAccepted = questAcceptedReward;
    out.questCompleted = questCompletedReward;
    out.questProgress = questProgressReward;
    out.questProximity = questProximityReward;

    float timePenalty = 0.001f;
    reward -= timePenalty;
    out.timePenalty = timePenalty;

    return reward;
}

void NeuralBotInstance::ResetRewardTracking()
{
    if (!_player) return;
    _prevXp = static_cast<float>(_player->GetUInt32Value(PLAYER_XP));
    _prevHealth = static_cast<float>(_player->GetHealth());
    _killCount = 0.0f;
    _diedThisStep = false;
    _stepCount = 0;
    _prevTrackedQuests.clear();
    // Prime with current quests so auto-accepted ones don't count as new each episode
    if (_player)
        for (uint32 slot = 0; slot < 25; ++slot)
        {
            uint32 qid = _player->GetQuestSlotQuestId(slot);
            if (qid)
                _prevTrackedQuests.insert(qid);
        }
    _prevQuestStatus.clear();
    _prevObjectiveCounts.clear();
    _cachedNearestQGDist = 0.0f;
    _prevQGDist = 0.0f;
    _questAutoCompleted = 0;
    _stepsWithoutReward = 0;
}

void NeuralBotInstance::ExecuteAction(uint32 action)
{
    Player* bot = _player;
    if (!bot || !bot->IsAlive())
        return;

    LOG_INFO("module.neuralbot", "Instance '{}' Action: {}", GetName(), action);

    switch (action)
    {
    case ACTION_NOOP:
        break;
    case ACTION_MOVE_FORWARD:
    {
        float dist = 3.0f;
        float dx, dy;
        if (Unit* sel = bot->GetSelectedUnit())
        {
            float angle = bot->GetAngle(sel);
            dx = cos(angle) * dist;
            dy = sin(angle) * dist;
        }
        else
        {
            float o = bot->GetOrientation();
            dx = cos(o) * dist;
            dy = std::sin(o) * dist;
        }
        float x = bot->GetPositionX() + dx;
        float y = bot->GetPositionY() + dy;
        float z = bot->GetPositionZ();
        bot->GetMotionMaster()->MovePoint(bot->GetMapId(), x, y, z, FORCED_MOVEMENT_RUN, false);
        break;
    }
    case ACTION_MOVE_BACKWARD:
    {
        float dist = 3.0f;
        float o = bot->GetOrientation();
        float dx = -cos(o) * dist;
        float dy = -std::sin(o) * dist;
        float x = bot->GetPositionX() + dx;
        float y = bot->GetPositionY() + dy;
        float z = bot->GetPositionZ();
        bot->GetMotionMaster()->MovePoint(bot->GetMapId(), x, y, z, FORCED_MOVEMENT_RUN, false);
        break;
    }
    case ACTION_TURN_LEFT:
    {
        float o = bot->GetOrientation() + 0.3f;
        if (o > 2.0f * static_cast<float>(M_PI))
            o -= 2.0f * static_cast<float>(M_PI);
        bot->SetOrientation(o);
        break;
    }
    case ACTION_TURN_RIGHT:
    {
        float o = bot->GetOrientation() - 0.3f;
        if (o < 0.0f)
            o += 2.0f * static_cast<float>(M_PI);
        bot->SetOrientation(o);
        break;
    }
    case ACTION_STOP_MOVE:
        bot->GetMotionMaster()->Clear();
        break;
    case ACTION_TARGET_NEAREST_ENEMY:
    {
        if (bot->IsInWorld())
        {
            float range = 40.0f;
            std::list<Unit*> targets;
            Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, range);
            Acore::UnitListSearcher<decltype(check)> searcher(bot, targets, check);
            Cell::VisitObjects(bot, searcher, range);
            Unit* nearest = nullptr;
            float minDist = range + 1.0f;
            for (Unit* u : targets)
            {
                if (!u || !u->IsAlive()) continue;
                float d = bot->GetDistance(u);
                if (d < minDist) { minDist = d; nearest = u; }
            }
            if (nearest)
                InjectCMSG(CMSG_SET_SELECTION, [nearest](WorldPacket& pkt) { pkt << nearest->GetGUID(); });
        }
        break;
    }
    case ACTION_ATTACK_START:
    {
        if (Unit* target = bot->GetSelectedUnit())
            InjectCMSG(CMSG_ATTACKSWING, [target](WorldPacket& pkt) { pkt << target->GetGUID(); });
        break;
    }
    case ACTION_CAST_SPELL_1:
    case ACTION_CAST_SPELL_2:
    case ACTION_CAST_SPELL_3:
    {
        size_t slot = action - ACTION_CAST_SPELL_1;
        uint32 spellId = _spellSlots[slot];
        if (spellId > 0 && bot->IsAlive())
        {
            Unit* target = bot->GetSelectedUnit();
            if (!target)
            {
                float range = 40.0f;
                std::list<Unit*> targets;
                Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, range);
                Acore::UnitListSearcher<decltype(check)> searcher(bot, targets, check);
                Cell::VisitObjects(bot, searcher, range);
                Unit* nearest = nullptr;
                float minDist = range + 1.0f;
                for (Unit* u : targets)
                {
                    if (!u || !u->IsAlive()) continue;
                    float d = bot->GetDistance(u);
                    if (d < minDist) { minDist = d; nearest = u; }
                }
                if (nearest) target = nearest;
            }
            InjectCMSG(CMSG_CAST_SPELL, [spellId, target](WorldPacket& pkt) {
                pkt << uint8(0);
                pkt << spellId;
                pkt << uint8(0);
                if (target)
                {
                    pkt << uint32(2);
                    pkt << target->GetGUID().WriteAsPacked();
                }
                else
                    pkt << uint32(0);
            });
        }
        break;
    }
    case ACTION_INTERACT_NPC:
    {
        Unit* target = bot->GetSelectedUnit();
        if (target && target->ToCreature())
        {
            ObjectGuid guid = target->GetGUID();
            Creature* creature = target->ToCreature();
            float dist = bot->GetDistance(creature);
            if (dist > 10.0f)
                LOG_INFO("module.neuralbot", "INTERACT_NPC too far: dist={} entry={}", dist, creature->GetEntry());
            InjectCMSG(CMSG_QUESTGIVER_HELLO, [guid](WorldPacket& pkt) { pkt << guid; });
            auto bounds = sObjectMgr->GetCreatureQuestRelationBounds(creature->GetEntry());
            for (auto it = bounds.first; it != bounds.second; ++it)
            {
                Quest const* quest = sObjectMgr->GetQuestTemplate(it->second);
                if (quest && bot->CanTakeQuest(quest, false))
                    InjectCMSG(CMSG_QUESTGIVER_ACCEPT_QUEST, [guid, questId = it->second](WorldPacket& pkt) {
                        pkt << guid << uint32(questId) << uint32(0);
                    });
            }
        }
        break;
    }
    case ACTION_COMPLETE_QUEST:
    {
        if (!bot->IsInWorld()) break;
        float range = 40.0f;
        std::list<Unit*> friendlies;
        Acore::AnyFriendlyUnitInObjectRangeCheck friendlyCheck(bot, bot, range);
        Acore::UnitListSearcher<decltype(friendlyCheck)> searcher(bot, friendlies, friendlyCheck);
        Cell::VisitObjects(bot, searcher, range);
        bool completed = false;
        for (uint32 slot = 0; slot < 25 && !completed; ++slot)
        {
            uint32 qid = bot->GetQuestSlotQuestId(slot);
            if (!qid || bot->GetQuestStatus(qid) != QUEST_STATUS_COMPLETE) continue;
            Quest const* quest = sObjectMgr->GetQuestTemplate(qid);
            if (!quest) continue;
            for (Unit* u : friendlies)
            {
                Creature* c = u->ToCreature();
                if (!c) continue;
                auto bounds = sObjectMgr->GetCreatureQuestInvolvedRelationBounds(c->GetEntry());
                for (auto it = bounds.first; it != bounds.second; ++it)
                {
                    if (it->second == qid)
                    {
                        ObjectGuid guid = c->GetGUID();
                        InjectCMSG(CMSG_QUESTGIVER_COMPLETE_QUEST, [guid, qid](WorldPacket& pkt) { pkt << guid << uint32(qid); });
                        InjectCMSG(CMSG_QUESTGIVER_CHOOSE_REWARD, [guid, qid](WorldPacket& pkt) { pkt << guid << uint32(qid) << uint32(0); });
                        completed = true;
                        break;
                    }
                }
                if (completed) break;
            }
        }
        break;
    }
    case ACTION_TARGET_QUEST_GIVER:
    {
        if (!bot->IsInWorld()) break;
        float range = 40.0f;
        std::list<Unit*> friendlies;
        Acore::AnyFriendlyUnitInObjectRangeCheck friendlyCheck(bot, bot, range);
        Acore::UnitListSearcher<decltype(friendlyCheck)> searcher(bot, friendlies, friendlyCheck);
        Cell::VisitObjects(bot, searcher, range);
        Unit* best = nullptr;
        float bestDist = range + 1.0f;
        for (Unit* u : friendlies)
        {
            Creature* c = u->ToCreature();
            if (!c) continue;
            uint32 entry = c->GetEntry();
            bool hasQuest = sObjectMgr->GetCreatureQuestRelationBounds(entry).first !=
                             sObjectMgr->GetCreatureQuestRelationBounds(entry).second;
            bool hasEnd = sObjectMgr->GetCreatureQuestInvolvedRelationBounds(entry).first !=
                          sObjectMgr->GetCreatureQuestInvolvedRelationBounds(entry).second;
            if (hasQuest || hasEnd)
            {
                float d = bot->GetDistance(c);
                if (d < bestDist) { bestDist = d; best = c; }
            }
        }
        if (best)
            InjectCMSG(CMSG_SET_SELECTION, [best](WorldPacket& pkt) { pkt << best->GetGUID(); });
        break;
    }
    }
}

NeuralBotStepResult NeuralBotInstance::Step(uint32 action)
{
    NeuralBotStepResult result;
    if (!_ready || !_player || !_player->IsAlive())
    {
        result.done = true;
        result.info = "Instance not ready or dead";
        return result;
    }

    ExecuteAction(action);
    _stepCount++;

    // Auto-complete quests if enabled (agent still gets reward signal)
    if (_autoQuestEnabled)
        AutoCompleteQuests();

    result.reward.total = ComputeReward(result.reward);
    BuildObservationInto(result.observation);

    // Idle termination: if agent hasn't earned reward in 50 steps, end episode
    if (result.reward.total > 0.01f)
        _stepsWithoutReward = 0;
    else
        _stepsWithoutReward++;

    bool timedOut = _stepCount >= _maxSteps;
    bool idle = _stepsWithoutReward >= 50;

    result.done = timedOut || _diedThisStep || !_player->IsAlive() || idle;
    if (result.done)
        result.info = _diedThisStep ? "died" : (idle ? "idle" : "max_steps");
    return result;
}

void NeuralBotInstance::AutoAcceptQuests()
{
    Player* bot = _player;
    if (!bot || !bot->IsInWorld()) return;

    float range = 20.0f;
    std::list<Unit*> friendlies;
    Acore::AnyFriendlyUnitInObjectRangeCheck check(bot, bot, range);
    Acore::UnitListSearcher<decltype(check)> searcher(bot, friendlies, check);
    Cell::VisitObjects(bot, searcher, range);

    uint32 accepted = 0;
    for (Unit* u : friendlies)
    {
        Creature* c = u->ToCreature();
        if (!c) continue;
        auto bounds = sObjectMgr->GetCreatureQuestRelationBounds(c->GetEntry());
        for (auto it = bounds.first; it != bounds.second; ++it)
        {
            uint32 questId = it->second;
            Quest const* quest = sObjectMgr->GetQuestTemplate(questId);
            if (!quest) continue;
            if (bot->CanTakeQuest(quest, false) && bot->GetQuestStatus(questId) == QUEST_STATUS_NONE)
            {
                bot->AddQuest(quest, c);
                accepted++;
                LOG_INFO("module.neuralbot", "Bot '{}' auto-accepted quest {} ({}): {}",
                    bot->GetName(), questId, quest->GetTitle().c_str(), quest->GetTitle().c_str());
            }
        }
    }
    if (accepted > 0)
        LOG_INFO("module.neuralbot", "Bot '{}' auto-accepted {} quest(s)", bot->GetName(), accepted);
}

void NeuralBotInstance::AutoCompleteQuests()
{
    Player* bot = _player;
    if (!bot) return;

    for (uint32 slot = 0; slot < 25; ++slot)
    {
        uint32 qid = bot->GetQuestSlotQuestId(slot);
        if (!qid) continue;
        if (bot->GetQuestStatus(qid) == QUEST_STATUS_COMPLETE)
        {
            Quest const* quest = sObjectMgr->GetQuestTemplate(qid);
            if (!quest) continue;
            bot->RewardQuest(quest, 0, bot, false);
            _questAutoCompleted++;
            LOG_INFO("module.neuralbot", "Bot '{}' auto-completed quest {} ({})",
                bot->GetName(), qid, quest->GetTitle().c_str());
            break; // Only one per step
        }
    }
}

NeuralBotObservation NeuralBotInstance::Reset()
{
    if (_player)
        ResetRewardTracking();
    NeuralBotObservation obs;
    BuildObservationInto(obs);
    return obs;
}
