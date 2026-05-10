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
        auto opcode = static_cast<OpcodeClient>(packet->GetOpcode());
        auto const* opHandle = opcodeTable[opcode];
        if (opHandle)
            opHandle->Call(_botSession, *packet);
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
            InjectCMSG(CMSG_CAST_SPELL, [spellId](WorldPacket& pkt) {
                pkt << uint8(0);
                pkt << spellId;
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
