#include "NeuralBotMgr.h"
#include "NeuralBotCommon.h"

#include "CharacterCache.h"
#include "Config.h"
#include "DatabaseEnv.h"
#include "DBCStructure.h"
#include "GridNotifiers.h"
#include "GridNotifiersImpl.h"
#include "Cell.h"
#include "CellImpl.h"
#include "Log.h"
#include "Map.h"
#include "MotionMaster.h"
#include "ObjectMgr.h"
#include "Player.h"
#include "ScriptMgr.h"
#include "SharedDefines.h"
#include "SpellMgr.h"
#include "Unit.h"
#include "World.h"
#include "WorldSession.h"
#include "WorldSessionMgr.h"

#include <algorithm>
#include <cmath>

NeuralBotMgr& NeuralBotMgr::instance()
{
    static NeuralBotMgr instance;
    return instance;
}

void NeuralBotMgr::Initialize()
{
    _enabled = sConfigMgr->GetOption<bool>("NeuralBot.Enable", false);
    if (!_enabled)
        return;

    _botAccountId = sConfigMgr->GetOption<uint32>("NeuralBot.BotAccountId", 0);
    _botCharacterName = sConfigMgr->GetOption<std::string>("NeuralBot.BotCharacterName", "Neuralbot");
    _maxSteps = sConfigMgr->GetOption<uint32>("NeuralBot.MaxEpisodeSteps", 1000);
    _tickRateMs = sConfigMgr->GetOption<uint32>("NeuralBot.TickRateMs", 50);

    LOG_INFO("module.neuralbot", "NeuralBot initialized. Account: {} Character: {}",
        _botAccountId, _botCharacterName);
}

void NeuralBotMgr::ScheduleLogin()
{
    if (_loginScheduled || _botReady)
        return;

    _loginScheduled = true;

    // Resolve character GUID from DB
    std::string escapedName = _botCharacterName;
    CharacterDatabase.EscapeString(escapedName);

    QueryResult result = CharacterDatabase.Query(
        "SELECT guid, account FROM characters WHERE name = '{}'", escapedName);

    if (!result)
    {
        LOG_ERROR("module.neuralbot", "Character '{}' not found in database", _botCharacterName);
        _loginScheduled = false;
        return;
    }

    Field* fields = result->Fetch();
    _botGuidLow = fields[0].Get<uint32>();
    uint32 charAccountId = fields[1].Get<uint32>();

    if (_botAccountId == 0)
        _botAccountId = charAccountId;

    _botGuid = ObjectGuid::Create<HighGuid::Player>(_botGuidLow);

    LOG_INFO("module.neuralbot", "Found character '{}' GUID:{} Account:{}",
        _botCharacterName, _botGuidLow, _botAccountId);

    // Delay login by 5 seconds to let the world fully initialize
    _loginTimer = 5000;
}

void NeuralBotMgr::Shutdown()
{
    _enabled = false;
    _botReady = false;
}

Player* NeuralBotMgr::GetBotPlayer()
{
    return _botPlayer;
}

WorldSession* NeuralBotMgr::GetBotSession()
{
    return _botSession;
}

void NeuralBotMgr::SetSpellSlot(size_t index, uint32 spellId)
{
    if (index < 5)
        _spellSlots[index] = spellId;
}

uint32 NeuralBotMgr::GetSpellSlot(size_t index) const
{
    return index < 5 ? _spellSlots[index] : 0;
}

void NeuralBotMgr::GetSpellbook(std::vector<uint32>& spells)
{
    Player* bot = _botPlayer;
    if (!bot)
        return;

    spells.clear();
    PlayerSpellMap const& map = bot->GetSpellMap();
    for (PlayerSpellMap::const_iterator itr = map.begin(); itr != map.end(); ++itr)
    {
        if (itr->second->State == PLAYERSPELL_REMOVED)
            continue;
        spells.push_back(itr->first);
    }
}

void NeuralBotMgr::SetSpellSlots(std::vector<uint32> const& spells)
{
    std::fill(std::begin(_spellSlots), std::end(_spellSlots), 0u);
    for (size_t i = 0; i < 5 && i < spells.size(); ++i)
        _spellSlots[i] = spells[i];
}

void NeuralBotMgr::AutoPopulateSpellSlots()
{
    Player* bot = _botPlayer;
    if (!bot)
        return;

    PlayerSpellMap const& map = bot->GetSpellMap();
    std::vector<uint32> combatSpells;

    for (PlayerSpellMap::const_iterator itr = map.begin(); itr != map.end(); ++itr)
    {
        if (itr->second->State == PLAYERSPELL_REMOVED)
            continue;

        SpellInfo const* spellInfo = sSpellMgr->GetSpellInfo(itr->first);
        if (!spellInfo)
            continue;
        if (spellInfo->IsPassive())
            continue;
        if (spellInfo->HasAttribute(SPELL_ATTR0_DO_NOT_DISPLAY))
            continue;
        if (spellInfo->HasEffect(SPELL_EFFECT_APPLY_AURA))
            continue;
        if (!spellInfo->HasEffect(SPELL_EFFECT_WEAPON_DAMAGE) &&
            !spellInfo->HasEffect(SPELL_EFFECT_SCHOOL_DAMAGE) &&
            !spellInfo->HasEffect(SPELL_EFFECT_ATTACK) &&
            spellInfo->SpellFamilyName == 0)
            continue;
        if (bot->HasSpellCooldown(itr->first))
            continue;

        combatSpells.push_back(itr->first);
    }

    std::fill(std::begin(_spellSlots), std::end(_spellSlots), 0u);
    for (size_t i = 0; i < 5 && i < combatSpells.size(); ++i)
        _spellSlots[i] = combatSpells[i];
}

void NeuralBotMgr::OnPlayerLogin(Player* player)
{
    if (!_enabled || !player)
        return;

    if (player->GetSession()->IsBot())
    {
        std::string playerName = player->GetName();
        std::string configName = _botCharacterName;
        std::transform(playerName.begin(), playerName.end(), playerName.begin(), ::tolower);
        std::transform(configName.begin(), configName.end(), configName.begin(), ::tolower);

        if (playerName == configName)
        {
            _botPlayer = player;
            _botSession = player->GetSession();
            _botGuid = player->GetGUID();
            _botReady = true;
            _prevXp = static_cast<float>(player->GetUInt32Value(PLAYER_XP));
            _prevHealth = static_cast<float>(player->GetHealth());
            LOG_INFO("module.neuralbot", "NeuralBot player logged in: {} GUID: {}",
                player->GetName(), player->GetGUID().GetCounter());
            AutoPopulateSpellSlots();
        }
    }
}

void NeuralBotMgr::OnPlayerAfterUpdate(Player* player, uint32 diff)
{
    if (!_enabled || !_botReady || player != _botPlayer)
        return;

    ProcessBotPackets();
}

void NeuralBotMgr::OnWorldUpdate(uint32 diff)
{
    if (_loginScheduled && !_botReady && _loginTimer > 0)
    {
        if (diff >= _loginTimer)
            _loginTimer = 0;
        else
            _loginTimer -= diff;

        if (_loginTimer == 0)
            DoLogin();
    }
}

void NeuralBotMgr::DoLogin()
{
    if (_botReady)
        return;

    if (_botGuidLow == 0)
    {
        LOG_ERROR("module.neuralbot", "Cannot login: no character GUID");
        return;
    }

    // Check if already online
    Player* existing = ObjectAccessor::FindConnectedPlayer(_botGuid);
    if (existing)
    {
        LOG_INFO("module.neuralbot", "Character '{}' already online", _botCharacterName);
        _botPlayer = existing;
        _botSession = existing->GetSession();
        _botReady = true;
        ResetRewardTracking();
        return;
    }

    // Create a bot WorldSession (same pattern as RandomPlayerbotMgr)
    WorldSession* botSession =
        new WorldSession(
            _botAccountId,
            "",
            0x0,
            nullptr,
            SEC_PLAYER,
            EXPANSION_WRATH_OF_THE_LICH_KING,
            time_t(0),
            sWorld->GetDefaultDbcLocale(),
            0,
            false,
            false,
            0,
            true
        );

    // Create LoginQueryHolder and load character
    auto holder = std::make_shared<LoginQueryHolder>(_botAccountId, _botGuid);
    if (!holder->Initialize())
    {
        LOG_ERROR("module.neuralbot", "Failed to initialize LoginQueryHolder for '{}'", _botCharacterName);
        delete botSession;
        _loginScheduled = false;
        return;
    }

    // Execute async DB query, then complete login on callback
    sWorld->AddQueryHolderCallback(CharacterDatabase.DelayQueryHolder(holder))
        .AfterComplete([this, botSession](SQLQueryHolderBase const& queryHolder)
        {
            LoginQueryHolder const& lqh = static_cast<LoginQueryHolder const&>(queryHolder);

            botSession->HandlePlayerLoginFromDB(lqh);

            Player* bot = botSession->GetPlayer();
            if (!bot)
            {
                LOG_ERROR("module.neuralbot", "Failed to load character '{}' from DB", _botCharacterName);
                botSession->LogoutPlayer(true);
                delete botSession;
                _loginScheduled = false;
                return;
            }

            _botPlayer = bot;
            _botSession = botSession;
            _botReady = true;

            ResetRewardTracking();

            LOG_INFO("module.neuralbot", "NeuralBot '{}' logged in successfully. Level {} Zone {}",
                bot->GetName(), static_cast<int>(bot->GetLevel()), bot->GetZoneId());
        });
}

void NeuralBotMgr::HandleBotLogin()
{
}

void NeuralBotMgr::OnPlayerbotPacketSent(Player* player, WorldPacket const* packet)
{
    if (!_enabled || !player || player != _botPlayer || !packet)
        return;

    RecordOpcode(packet->GetOpcode());
}

void NeuralBotMgr::OnPlayerJustDied(Player* player)
{
    if (!_enabled || !player || player != _botPlayer)
        return;
    _diedThisStep = true;
}

void NeuralBotMgr::OnPlayerCreatureKill(Player* killer, Creature* killed)
{
    if (!_enabled || !killer || killer != _botPlayer)
        return;
    _killCount += 1.0f;
}

void NeuralBotMgr::RecordOpcode(uint16 opcode)
{
    std::lock_guard<std::mutex> lock(_opcodeMutex);
    _opcodeHistory.push_back(opcode);
    if (_opcodeHistory.size() > OBS_OPCODE_HISTORY_SIZE)
        _opcodeHistory.pop_front();
}

void NeuralBotMgr::ProcessBotPackets()
{
    if (!_botSession)
        return;

    WorldPacket* packet;
    auto& queue = _botSession->GetPacketQueue();
    while (queue.next(packet))
    {
        if (!packet)
            continue;
        try
        {
            auto opcode = static_cast<OpcodeClient>(packet->GetOpcode());
            auto const* opHandle = opcodeTable[opcode];
            if (opHandle)
                opHandle->Call(_botSession, *packet);
        }
        catch (ByteBufferException const& e)
        {
            auto opcode = static_cast<OpcodeClient>(packet->GetOpcode());
            LOG_ERROR("module.neuralbot", "ByteBufferException for opcode {}: {}", opcode, e.what());
        }
        delete packet;
    }
}

void NeuralBotMgr::InjectCMSG(uint16 opcode, std::function<void(WorldPacket&)> filler)
{
    if (!_botSession)
        return;

    WorldPacket* pkt = new WorldPacket(opcode, 64);
    if (filler)
        filler(*pkt);
    pkt->resize(pkt->wpos());
    _botSession->QueuePacket(pkt);
}

void NeuralBotMgr::BuildObservationInto(NeuralBotObservation& obs)
{
    Player* bot = _botPlayer;
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
        {
            if (u && u->IsInWorld())
                nearbyUnits.push_back({u, bot->GetDistance(u)});
        }

        std::sort(nearbyUnits.begin(), nearbyUnits.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
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

        // Scan quest log
        for (uint32 slot = 0; slot < 25; ++slot)
        {
            uint32 qid = bot->GetQuestSlotQuestId(slot);
            if (!qid)
                continue;

            questCount++;

            QuestStatus status = bot->GetQuestStatus(qid);
            if (status == QUEST_STATUS_COMPLETE)
                hasCompletable = true;

            Quest const* quest = sObjectMgr->GetQuestTemplate(qid);
            if (quest)
            {
                uint32 curSum = 0;
                uint32 reqSum = 0;
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

        // Scan friendly NPCs for quest givers/enders
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
                if (!c)
                    continue;

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

    // --- Opcode history ---
    {
        std::lock_guard<std::mutex> lock(_opcodeMutex);
        size_t histSize = _opcodeHistory.size();
        for (size_t i = 0; i < OBS_OPCODE_HISTORY_SIZE; ++i)
        {
            if (i < histSize)
                obs.opcodeHistory[i] = static_cast<int32>(_opcodeHistory[histSize - 1 - i]);
            else
                obs.opcodeHistory[i] = 0;
        }
    }
}

float NeuralBotMgr::ComputeReward()
{
    Player* bot = _botPlayer;
    if (!bot)
        return 0.0f;

    float reward = 0.0f;

    float curXp = static_cast<float>(bot->GetUInt32Value(PLAYER_XP));
    float xpDelta = curXp - _prevXp;
    reward += xpDelta / 100.0f;
    _prevXp = curXp;

    reward += _killCount * 1.0f;
    _killCount = 0.0f;

    float curHp = static_cast<float>(bot->GetHealth());
    float hpDelta = _prevHealth - curHp;
    if (hpDelta > 0)
        reward -= hpDelta / static_cast<float>(bot->GetMaxHealth()) * 0.5f;
    _prevHealth = curHp;

    if (_diedThisStep)
    {
        reward -= 10.0f;
        _diedThisStep = false;
    }

    // --- Quest rewards ---
    {
        // Snapshot current quest state
        std::set<uint32> curActive;
        std::map<uint32, uint8> curStatus;
        std::map<uint32, std::array<uint16, 4>> curCounters;

        for (uint32 slot = 0; slot < 25; ++slot)
        {
            uint32 qid = bot->GetQuestSlotQuestId(slot);
            if (!qid)
                continue;

            curActive.insert(qid);
            curStatus[qid] = static_cast<uint8>(bot->GetQuestStatus(qid));

            std::array<uint16, 4> counts = {
                bot->GetQuestSlotCounter(slot, 0),
                bot->GetQuestSlotCounter(slot, 1),
                bot->GetQuestSlotCounter(slot, 2),
                bot->GetQuestSlotCounter(slot, 3)
            };
            curCounters[qid] = counts;
        }

        // Check previously seen quests that may now be rewarded
        for (uint32 qid : _prevTrackedQuests)
            if (curActive.find(qid) == curActive.end())
                curStatus[qid] = static_cast<uint8>(bot->GetQuestStatus(qid));

        // Detect newly accepted quests
        for (uint32 qid : curActive)
            if (_prevTrackedQuests.find(qid) == _prevTrackedQuests.end())
                reward += 5.0f;

        // Detect newly completed/rewarded quests
        for (auto const& [qid, prevStatus] : _prevQuestStatus)
        {
            auto it = curStatus.find(qid);
            if (it != curStatus.end())
            {
                uint8 cur = it->second;
                if (prevStatus == static_cast<uint8>(QUEST_STATUS_COMPLETE) &&
                    cur == static_cast<uint8>(QUEST_STATUS_REWARDED))
                    reward += 20.0f;
            }
        }

        // Detect objective progress
        for (auto const& [qid, prevCounts] : _prevObjectiveCounts)
        {
            auto it = curCounters.find(qid);
            if (it != curCounters.end())
                for (int i = 0; i < 4; ++i)
                    if (it->second[i] > prevCounts[i])
                        reward += 0.5f * static_cast<float>(it->second[i] - prevCounts[i]);
        }

        // Reward proximity to quest giver
        if (_cachedNearestQGDist > 0.0f && _cachedNearestQGDist < 5.0f)
            reward += 0.5f * (1.0f - _cachedNearestQGDist / 5.0f);

        // Update snapshots
        _prevTrackedQuests.insert(curActive.begin(), curActive.end());
        _prevQuestStatus = std::move(curStatus);
        _prevObjectiveCounts = std::move(curCounters);
    }

    reward -= 0.001f;

    return reward;
}

void NeuralBotMgr::ResetRewardTracking()
{
    if (!_botPlayer)
        return;
    _prevXp = static_cast<float>(_botPlayer->GetUInt32Value(PLAYER_XP));
    _prevHealth = static_cast<float>(_botPlayer->GetHealth());
    _killCount = 0.0f;
    _diedThisStep = false;
    _stepCount = 0;

    _prevTrackedQuests.clear();
    _prevQuestStatus.clear();
    _prevObjectiveCounts.clear();
    _cachedNearestQGDist = 0.0f;
}

void NeuralBotMgr::ExecuteAction(uint32 action)
{
    Player* bot = _botPlayer;
    if (!bot || !bot->IsAlive())
        return;

    switch (action)
    {
    case ACTION_NOOP:
        break;
    case ACTION_MOVE_FORWARD:
    case ACTION_MOVE_BACKWARD:
    case ACTION_MOVE_LEFT:
    case ACTION_MOVE_RIGHT:
    case ACTION_MOVE_FORWARD_LEFT:
    case ACTION_MOVE_FORWARD_RIGHT:
    case ACTION_MOVE_BACKWARD_LEFT:
    case ACTION_MOVE_BACKWARD_RIGHT:
    {
        float dx = 0.0f, dy = 0.0f;
        float dist = 3.0f;
        float o = bot->GetOrientation();

        if (action == ACTION_MOVE_FORWARD || action == ACTION_MOVE_FORWARD_LEFT || action == ACTION_MOVE_FORWARD_RIGHT)
        {
            dx += cos(o) * dist;
            dy += std::sin(o) * dist;
        }
        if (action == ACTION_MOVE_BACKWARD || action == ACTION_MOVE_BACKWARD_LEFT || action == ACTION_MOVE_BACKWARD_RIGHT)
        {
            dx -= cos(o) * dist;
            dy -= std::sin(o) * dist;
        }
        if (action == ACTION_MOVE_LEFT || action == ACTION_MOVE_FORWARD_LEFT || action == ACTION_MOVE_BACKWARD_LEFT)
        {
            float strafeO = o + static_cast<float>(M_PI) * 0.5f;
            dx += cos(strafeO) * dist;
            dy += std::sin(strafeO) * dist;
        }
        if (action == ACTION_MOVE_RIGHT || action == ACTION_MOVE_FORWARD_RIGHT || action == ACTION_MOVE_BACKWARD_RIGHT)
        {
            float strafeO = o - static_cast<float>(M_PI) * 0.5f;
            dx += cos(strafeO) * dist;
            dy += std::sin(strafeO) * dist;
        }

        float x = bot->GetPositionX() + dx;
        float y = bot->GetPositionY() + dy;
        float z = bot->GetPositionZ();
        bot->GetMotionMaster()->MovePoint(bot->GetMapId(), x, y, z, FORCED_MOVEMENT_RUN, false);
        break;
    }
    case ACTION_STOP_MOVE:
        bot->GetMotionMaster()->Clear();
        break;
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
                if (!u || !u->IsAlive())
                    continue;
                float d = bot->GetDistance(u);
                if (d < minDist)
                {
                    minDist = d;
                    nearest = u;
                }
            }

            if (nearest)
            {
                InjectCMSG(CMSG_SET_SELECTION, [nearest](WorldPacket& pkt) {
                    pkt << nearest->GetGUID();
                });
            }
        }
        break;
    }
    case ACTION_TARGET_BY_INDEX_0:
    case ACTION_TARGET_BY_INDEX_1:
    case ACTION_TARGET_BY_INDEX_2:
    case ACTION_TARGET_BY_INDEX_3:
    {
        size_t idx = action - ACTION_TARGET_BY_INDEX_0;
        if (bot->IsInWorld())
        {
            float range = 40.0f;
            std::list<Unit*> targets;
            Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, range);
            Acore::UnitListSearcher<decltype(check)> searcher(bot, targets, check);
            Cell::VisitObjects(bot, searcher, range);

            std::vector<std::pair<Unit*, float>> sorted;
            for (Unit* u : targets)
            {
                if (u && u->IsAlive())
                    sorted.push_back({u, bot->GetDistance(u)});
            }
            std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            if (idx < sorted.size())
            {
                InjectCMSG(CMSG_SET_SELECTION, [sorted, idx](WorldPacket& pkt) {
                    pkt << sorted[idx].first->GetGUID();
                });
            }
        }
        break;
    }
    case ACTION_ATTACK_START:
    {
        if (Unit* target = bot->GetSelectedUnit())
        {
            InjectCMSG(CMSG_ATTACKSWING, [target](WorldPacket& pkt) {
                pkt << target->GetGUID();
            });
        }
        break;
    }
    case ACTION_ATTACK_STOP:
        InjectCMSG(CMSG_ATTACKSTOP, nullptr);
        break;
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
                    if (!u || !u->IsAlive())
                        continue;
                    float d = bot->GetDistance(u);
                    if (d < minDist)
                    {
                        minDist = d;
                        nearest = u;
                    }
                }

                if (nearest)
                    target = nearest;
            }

            InjectCMSG(CMSG_CAST_SPELL, [spellId, target](WorldPacket& pkt) {
                pkt << uint8(0);       // castCount
                pkt << spellId;        // spellId (uint32)
                pkt << uint8(0);       // castFlags

                if (target)
                {
                    pkt << uint32(2);              // targetMask = TARGET_FLAG_UNIT
                    pkt << target->GetGUID().WriteAsPacked();  // packed target guid
                }
                else
                {
                    pkt << uint32(0);              // targetMask = TARGET_FLAG_NONE
                }
            });
        }
        break;
    }
    case ACTION_INTERACT_LOOT:
    {
        if (Unit* target = bot->GetSelectedUnit())
        {
            if (Creature* creature = target->ToCreature())
            {
                if (creature->isDead())
                {
                    InjectCMSG(CMSG_LOOT, [creature](WorldPacket& pkt) {
                        pkt << creature->GetGUID();
                    });
                }
            }
        }
        break;
    }
    case ACTION_STAND_UP:
        bot->SetStandState(UNIT_STAND_STATE_STAND);
        break;
    case ACTION_INTERACT_NPC:
    {
        Unit* target = bot->GetSelectedUnit();
        if (target && target->ToCreature())
        {
            ObjectGuid guid = target->GetGUID();
            Creature* creature = target->ToCreature();

            InjectCMSG(CMSG_QUESTGIVER_HELLO, [guid](WorldPacket& pkt) {
                pkt << guid;
            });

            auto bounds = sObjectMgr->GetCreatureQuestRelationBounds(creature->GetEntry());
            for (auto it = bounds.first; it != bounds.second; ++it)
            {
                Quest const* quest = sObjectMgr->GetQuestTemplate(it->second);
                if (quest && bot->CanTakeQuest(quest, false))
                {
                    uint32 questId = it->second;
                    InjectCMSG(CMSG_QUESTGIVER_ACCEPT_QUEST, [guid, questId](WorldPacket& pkt) {
                        pkt << guid;
                        pkt << uint32(questId);
                        pkt << uint32(0);
                    });
                }
            }
        }
        break;
    }
    case ACTION_COMPLETE_QUEST:
    {
        if (!bot->IsInWorld())
            break;

        float range = 40.0f;
        std::list<Unit*> friendlies;
        Acore::AnyFriendlyUnitInObjectRangeCheck friendlyCheck(bot, bot, range);
        Acore::UnitListSearcher<decltype(friendlyCheck)> searcher(bot, friendlies, friendlyCheck);
        Cell::VisitObjects(bot, searcher, range);

        bool completed = false;
        for (uint32 slot = 0; slot < 25 && !completed; ++slot)
        {
            uint32 qid = bot->GetQuestSlotQuestId(slot);
            if (!qid)
                continue;

            if (bot->GetQuestStatus(qid) != QUEST_STATUS_COMPLETE)
                continue;

            Quest const* quest = sObjectMgr->GetQuestTemplate(qid);
            if (!quest)
                continue;

            for (Unit* u : friendlies)
            {
                Creature* c = u->ToCreature();
                if (!c)
                    continue;

                auto bounds = sObjectMgr->GetCreatureQuestInvolvedRelationBounds(c->GetEntry());
                for (auto it = bounds.first; it != bounds.second; ++it)
                {
                    if (it->second == qid)
                    {
                        ObjectGuid guid = c->GetGUID();
                        InjectCMSG(CMSG_QUESTGIVER_COMPLETE_QUEST, [guid, qid](WorldPacket& pkt) {
                            pkt << guid;
                            pkt << uint32(qid);
                        });
                        InjectCMSG(CMSG_QUESTGIVER_CHOOSE_REWARD, [guid, qid](WorldPacket& pkt) {
                            pkt << guid;
                            pkt << uint32(qid);
                            pkt << uint32(0);
                        });
                        completed = true;
                        break;
                    }
                }
                if (completed)
                    break;
            }
        }
        break;
    }
    case ACTION_TARGET_QUEST_GIVER:
    {
        if (!bot->IsInWorld())
            break;

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
            if (!c)
                continue;

            uint32 entry = c->GetEntry();
            auto startBounds = sObjectMgr->GetCreatureQuestRelationBounds(entry);
            auto endBounds = sObjectMgr->GetCreatureQuestInvolvedRelationBounds(entry);

            if (startBounds.first != startBounds.second || endBounds.first != endBounds.second)
            {
                float d = bot->GetDistance(c);
                if (d < bestDist)
                {
                    bestDist = d;
                    best = c;
                }
            }
        }

        if (best)
        {
            InjectCMSG(CMSG_SET_SELECTION, [best](WorldPacket& pkt) {
                pkt << best->GetGUID();
            });
        }
        break;
    }
    }
}

NeuralBotStepResult NeuralBotMgr::Step(uint32 action)
{
    NeuralBotStepResult result;

    if (!_botReady || !_botPlayer || !_botPlayer->IsAlive())
    {
        result.done = true;
        result.info = "Bot not ready or dead";
        return result;
    }

    ExecuteAction(action);
    _stepCount++;

    result.reward.total = ComputeReward();

    BuildObservationInto(result.observation);

    result.done = (_stepCount >= _maxSteps) || _diedThisStep || !_botPlayer->IsAlive();
    if (result.done)
        result.info = _diedThisStep ? "died" : "max_steps";

    return result;
}

NeuralBotObservation NeuralBotMgr::Reset()
{
    if (_botPlayer)
    {
        ResetRewardTracking();
    }

    NeuralBotObservation obs;
    BuildObservationInto(obs);
    return obs;
}
