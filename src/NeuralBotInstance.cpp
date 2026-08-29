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
#include "ObjectAccessor.h"
#include "Trainer.h"
#include "GameObject.h"

#include <algorithm>
#include <cmath>

namespace
{
// All units (any faction, incl. corpses) in range — used for the faithful entity list.
struct AnyUnitInRangeCheck
{
    AnyUnitInRangeCheck(WorldObject const* obj, float range) : i_obj(obj), i_range(range) {}
    bool operator()(Unit* u)
    {
        return u && u->IsInWorld() && u != i_obj && i_obj->IsWithinDistInMap(u, i_range);
    }
    WorldObject const* i_obj;
    float i_range;
};

struct AnyGameObjectInRangeCheck
{
    AnyGameObjectInRangeCheck(WorldObject const* obj, float range) : i_obj(obj), i_range(range) {}
    bool operator()(GameObject* go)
    {
        return go && go->IsInWorld() && i_obj->IsWithinDistInMap(go, i_range);
    }
    WorldObject const* i_obj;
    float i_range;
};

uint8 ComputeReaction(Unit const* bot, Unit const* u)
{
    if (u->IsFriendlyTo(bot))
        return NB_REACTION_FRIENDLY;
    if (u->IsHostileTo(bot))
        return NB_REACTION_HOSTILE;
    return NB_REACTION_NEUTRAL;
}

uint8 UnitEntityType(Unit const* u)
{
    return u->IsPlayer() ? NB_ENTITY_TYPE_PLAYER : NB_ENTITY_TYPE_CREATURE;
}
} // namespace

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

void NeuralBotInstance::ReviveIfDead()
{
    Player* bot = _player;
    if (!bot || bot->IsAlive())
        return;

    // Nothing in the environment ever revives a bot: a corpse stays a corpse, and
    // ShouldTerminate() then fires every step (!IsAlive), turning that env slot into an
    // endless stream of length-1 zero-reward episodes. Revive at the death spot with 50%
    // health (same sequence mod-playerbots uses at spirit healers) so the next episode
    // starts immediately.
    bot->ResurrectPlayer(0.5f);
    bot->SpawnCorpseBones();
    bot->SetTarget();
    _diedThisStep = false;
    _stepsWithoutReward = 0;
}

namespace
{
struct LevelBandHub
{
    uint16 mapId;
    float x, y, z;
};

struct HubList
{
    uint8 count;
    LevelBandHub hubs[4];
};

// Level-band respawn hubs — mob-cluster centroids (NOT towns/graveyards: those sit
// 100-300yd from hostile fields, beyond the 100yd sense range, stranding bots).
// Computed from creature x factiontemplate_dbc where (FriendGroup & 7)==0 (not friendly
// to any player group) and (FactionGroup & 8)!=0 (monster), averaged over the zone's
// level-appropriate spawns. Bands: 0 = 1-4, 1 = 5-9, 2 = 10-19. Level 20+ clamps to
// band 2 for now.
HubList const& GetLevelBandHubs(uint8 team, uint8 band)
{
    static HubList const ALLIANCE[3] = {
        // band 0: 1-4 — mob-cluster centroids (computed from factiontemplate_dbc:
        // (FriendGroup & 7)==0 and (FactionGroup & 8)!=0, minlevel 1-5), NOT graveyards.
        // Graveyards sit ~100-200yd from hostile spawns (nearest are critters/friendly
        // NPCs), so lost bots respawned there still saw no mobs. These land them in the
        // middle of the level-1-4 killable population.
        { 4, { { 0,   -8830.0f,  -111.0f,   83.0f },   // Elwynn, Northshire (kobolds)
                { 0,   -5977.0f,    99.0f,  395.0f },   // Dun Morogh, Coldridge (troggs)
                { 1,    9914.0f,   857.0f, 1309.0f },   // Teldrassil, Shadowglen
                { 530, -3794.0f, -13328.0f,  72.0f } } }, // Azuremyst, Ammen Vale
        // band 1: 5-9 — mob-cluster centroids (same faction filter, minlevel 5-9)
        { 4, { { 0,   -9569.0f,     92.0f,   51.0f },   // Elwynn (defias/murlocs)
                { 0,   -5435.0f,   -180.0f,  391.0f },   // Dun Morogh (leopards/trolls)
                { 1,    9792.0f,   1259.0f, 1304.0f },   // Teldrassil (owlbeasts/spiders)
                { 530, -4201.0f, -12071.0f,    9.0f } } }, // Azuremyst (stags/moonstalkers)
        // band 2: 10-19 — mob-cluster centroids (minlevel 10-15)
        { 4, { { 0,  -10343.0f,   1538.0f,   32.0f },   // Westfall (defias harvest)
                { 0,   -5373.0f,  -3025.0f,  341.0f },   // Loch Modan (troggs/spiders)
                { 1,    6684.0f,     88.0f,   20.0f },   // Darkshore (crawlers/furbolgs)
                { 530, -2048.0f, -11827.0f,   19.0f } } }, // Bloodmyst (ravagers/treants)
    };
    static HubList const HORDE[3] = {
        // band 0: 1-4 — mob-cluster centroids (see ALLIANCE comment).
        { 4, { { 1,   -653.0f,  -4656.0f,  39.0f },   // Durotar, Valley of Trials
                { 0,   2302.0f,   1329.0f,  35.0f },   // Tirisfal, Deathknell
                { 1,  -2635.0f,    -90.0f,  27.0f },   // Mulgore, Red Cloud Mesa (Narache)
                { 530, 9840.0f,  -6707.0f,  11.0f } } }, // Eversong, Sunstrider Isle
        // band 1: 5-9 — mob-cluster centroids (see ALLIANCE comment)
        { 4, { { 1,     420.0f,  -4461.0f,   25.0f },   // Durotar (scorpids/quillboar)
                { 0,    2480.0f,    557.0f,   46.0f },   // Tirisfal (scarlet/darkhounds)
                { 1,   -1909.0f,   -505.0f,   27.0f },   // Mulgore (plainstriders/wolves)
                { 530,  8740.0f,  -6555.0f,   46.0f } } }, // Eversong (wyrms/wraiths)
        // band 2: 10-19 — mob-cluster centroids (minlevel 10-15)
        { 3, { { 1,    -441.0f,  -2448.0f,   99.0f },   // Barrens (razormanes/striders)
                { 0,     889.0f,   1321.0f,   49.0f },   // Silverpine (worgen/wraiths)
                { 530,  7477.0f,  -6313.0f,   27.0f },   // Ghostlands (bats/ghostcallers)
                { 0,      0.0f,      0.0f,   0.0f } } }, // unused
    };
    return (team == 0) ? ALLIANCE[band] : HORDE[band];
}
} // namespace

void NeuralBotInstance::RespawnToLevelBand()
{
    Player* bot = _player;
    if (!bot || !bot->IsInWorld() || !bot->IsAlive())
        return;

    uint32 level = bot->GetLevel();
    uint8 band;
    if (level >= 10)
        band = 2;
    else if (level >= 5)
        band = 1;
    else
    {
        // Level 1-4: only respawn when LOST. 87% of level-1 bots random-walk into
        // empty areas (0 entities in obs) where they have no learning signal — this
        // is the real kill/XP bottleneck. Rescue them by teleporting ~12yd from the
        // nearest hostile (wide search); if truly none, fall back to the graveyard.
        float range = 300.0f;
        std::list<Unit*> units;
        Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, range);
        Acore::UnitListSearcher<decltype(check)> searcher(bot, units, check);
        Cell::VisitObjects(bot, searcher, range);
        Unit* nearest = nullptr;
        float bestDist = range + 1.0f;
        for (Unit* u : units)
        {
            if (!u || !u->IsAlive() || u->IsCritter())
                continue;
            float d = bot->GetDistance(u);
            if (d < bestDist) { bestDist = d; nearest = u; }
        }
        if (nearest)
        {
            float a = frand(0.0f, 6.2831853f);
            bot->GetMotionMaster()->Clear();
            bot->TeleportTo(bot->GetMapId(),
                nearest->GetPositionX() + std::cos(a) * 12.0f,
                nearest->GetPositionY() + std::sin(a) * 12.0f,
                nearest->GetPositionZ(), frand(0.0f, 6.2831853f));
            LOG_DEBUG("module.neuralbot", "'{}' rescued near hostile '{}' ({}yd away, {} units scanned)", GetName(), nearest->GetName(), bestDist, units.size());
            return;
        }
        LOG_DEBUG("module.neuralbot", "'{}' LOST: no hostile within {}yd ({} units scanned) — graveyard fallback", GetName(), range, units.size());
        band = 0;
    }
    uint8 team = (bot->GetTeamId() == TEAM_ALLIANCE) ? 0 : 1;
    HubList const& list = GetLevelBandHubs(team, band);
    if (list.count == 0)
        return;

    LevelBandHub const& hub = list.hubs[urand(0, list.count - 1)];
    float jitter = frand(-3.0f, 3.0f);
    bot->GetMotionMaster()->Clear();
    bot->TeleportTo(hub.mapId, hub.x + jitter, hub.y + jitter, hub.z, frand(0.0f, 6.2831853f));
    LOG_DEBUG("module.neuralbot", "'{}' respawned to level-band hub (level {}, map {})",
        GetName(), level, hub.mapId);
}

void NeuralBotInstance::StageEpisodeStart()
{
    // Curriculum staging (ROADMAP §7): when a bot starts an episode too far from any
    // hostile to reach it within a reasonable fraction of the episode, teleport it to
    // ~12 yd of the nearest one. This collapses the ~190-step approach phase that made
    // the native XP reward effectively unreachable for credit assignment; rewards stay
    // 100% native — only the starting distribution changes. Once the policy fights
    // reliably, staging is disabled (config) and spawn positions restore the challenge.
    Player* bot = _player;
    if (!bot || !bot->IsAlive() || !bot->IsInWorld() || bot->IsInCombat())
        return;

    float const scanRange = 120.0f;
    std::list<Unit*> units;
    Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, scanRange);
    Acore::UnitListSearcher<decltype(check)> searcher(bot, units, check);
    Cell::VisitObjects(bot, searcher, scanRange);

    Unit* nearest = nullptr;
    float bestDist = scanRange + 1.0f;
    for (Unit* u : units)
    {
        if (!u || !u->IsAlive() || u->IsCritter())
            continue;
        float d = bot->GetDistance(u);
        if (d < bestDist)
        {
            bestDist = d;
            nearest = u;
        }
    }

    // Already in reach of a fight — nothing to stage.
    if (bestDist <= 60.0f)
        return;
    if (!nearest)
        return;

    // Touchdown point ~12 yd from the mob, approached from the bot's current side.
    float dx = bot->GetPositionX() - nearest->GetPositionX();
    float dy = bot->GetPositionY() - nearest->GetPositionY();
    float len = std::sqrt(dx * dx + dy * dy);
    if (len < 0.1f)
    {
        dx = 1.0f;
        dy = 0.0f;
        len = 1.0f;
    }
    float x = nearest->GetPositionX() + dx / len * 12.0f;
    float y = nearest->GetPositionY() + dy / len * 12.0f;
    float z = nearest->GetPositionZ();
    float o = nearest->GetAngle(bot);

    bot->GetMotionMaster()->Clear();
    bot->NearTeleportTo(x, y, z, o);
    LOG_DEBUG("module.neuralbot", "'{}' staged to fight (was {:.0f} yd out)", GetName(), bestDist);
}

void NeuralBotInstance::OnPlayerCreatureKill(Creature* killed)
{
    _killCount += 1.0f;
    _killsThisEpisode++;
    if (killed)
        _lastKilledGuid = killed->GetGUID();
}

void NeuralBotInstance::OnPlayerLearnSpell(uint32 /*spellId*/)
{
    _spellsLearnedThisEpisode++;
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
        float range = NB_SENSE_RANGE;
        std::list<Unit*> targets;
        Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, range);
        Acore::UnitListSearcher<decltype(check)> searcher(bot, targets, check);
        Cell::VisitObjects(bot, searcher, range);

        for (Unit* u : targets)
            if (u && u->IsInWorld())
                nearbyUnits.push_back({u, bot->GetDistance(u)});

        std::sort(nearbyUnits.begin(), nearbyUnits.end(),
            [](auto const& a, auto const& b) { return a.second < b.second; });

        _cachedNearestEnemyDist = nearbyUnits.empty() ? 0.0f : nearbyUnits[0].second;
    }
    else
        _cachedNearestEnemyDist = 0.0f;

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

    // --- Trainer state ---
    {
        float nearestTrainerDist = 0.0f;
        uint32 learnableCount = 0;
        bool canAfford = false;
        Unit* nearestTrainer = nullptr;

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
                if (!c || !(c->GetNpcFlags() & UNIT_NPC_FLAG_TRAINER)) continue;
                Trainer::Trainer const* trainer = sObjectMgr->GetTrainer(c->GetEntry());
                if (!trainer || !trainer->IsTrainerValidForPlayer(bot)) continue;

                float dist = bot->GetDistance(c);
                if (!nearestTrainer || dist < nearestTrainerDist)
                    nearestTrainerDist = dist;

                if (!nearestTrainer)
                {
                    nearestTrainer = c;
                    for (auto const& spell : trainer->GetSpells())
                        if (trainer->CanTeachSpell(bot, &spell))
                            learnableCount++;
                    if (learnableCount > 0 && !trainer->GetSpells().empty()
                        && trainer->GetSpells()[0].MoneyCost <= bot->GetMoney())
                        canAfford = true;
                }
            }
        }

        _cachedNearestTrainerDist = nearestTrainerDist;
        obs.combatState[10] = nearestTrainerDist > 0.0f ? std::min(nearestTrainerDist / 40.0f, 1.0f) : 1.0f;
        obs.combatState[11] = (nearestTrainerDist > 0.0f && nearestTrainerDist <= 5.0f) ? 1.0f : 0.0f;
        obs.combatState[12] = std::min(static_cast<float>(learnableCount) / 10.0f, 1.0f);
        obs.combatState[13] = canAfford ? 1.0f : 0.0f;
        obs.combatState[14] = (nearestTrainer != nullptr) ? 1.0f : 0.0f;
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

void BuildFrameFor(Player* bot, NeuralBotFrame& frame, ObjectGuid* entityGuidsOut, size_t* entityCountOut)
{
    if (!bot)
        return;

    std::memset(&frame, 0, sizeof(NeuralBotFrame));

    // ── self ────────────────────────────────────────────────────────────
    NBStateSelf& self = frame.self;
    self.guid = bot->GetGUID().GetRawValue();
    self.level = bot->GetLevel();
    self.health = static_cast<float>(bot->GetHealth());
    self.maxHealth = static_cast<float>(bot->GetMaxHealth());
    self.mana = static_cast<float>(bot->GetPower(POWER_MANA));
    self.maxMana = static_cast<float>(bot->GetMaxPower(POWER_MANA));
    Powers powerType = bot->getPowerType();
    if (powerType != POWER_MANA)
    {
        self.resource = static_cast<float>(bot->GetPower(powerType));
        self.maxResource = static_cast<float>(bot->GetMaxPower(powerType));
    }
    self.xp = bot->GetUInt32Value(PLAYER_XP);
    self.nextLevelXp = bot->GetUInt32Value(PLAYER_NEXT_LEVEL_XP);
    self.money = bot->GetMoney();
    self.posX = bot->GetPositionX();
    self.posY = bot->GetPositionY();
    self.posZ = bot->GetPositionZ();
    self.orientation = bot->GetOrientation();
    self.mapId = bot->GetMapId();
    self.zoneId = bot->GetZoneId();
    self.areaId = bot->GetAreaId();
    self.alive = bot->IsAlive() ? 1 : 0;
    self.inCombat = bot->IsInCombat() ? 1 : 0;
    self.moving = bot->isMoving() ? 1 : 0;
    self.casting = bot->HasUnitState(UNIT_STATE_CASTING) ? 1 : 0;
    self.inWater = bot->IsInWater() ? 1 : 0;
    self.mounted = bot->IsMounted() ? 1 : 0;
    self.classId = bot->getClass();
    self.race = bot->getRace();
    self.comboPoints = bot->GetComboPoints();

    // ── target ──────────────────────────────────────────────────────────
    if (Unit* target = bot->GetSelectedUnit())
    {
        NBStateTarget& t = frame.target;
        t.guid = target->GetGUID().GetRawValue();
        t.entry = target->GetEntry();
        t.type = UnitEntityType(target);
        t.health = static_cast<float>(target->GetHealth());
        t.maxHealth = static_cast<float>(target->GetMaxHealth());
        t.level = target->GetLevel();
        t.dx = target->GetPositionX() - bot->GetPositionX();
        t.dy = target->GetPositionY() - bot->GetPositionY();
        t.dz = target->GetPositionZ() - bot->GetPositionZ();
        t.distance = bot->GetDistance(target);
        t.reaction = ComputeReaction(bot, target);
        t.alive = target->IsAlive() ? 1 : 0;
        t.inCombat = target->IsInCombat() ? 1 : 0;
        t.casting = target->HasUnitState(UNIT_STATE_CASTING) ? 1 : 0;
        t.npcFlags = target->ToCreature() ? static_cast<uint32_t>(target->GetNpcFlags()) : 0;
        self.targetGuid = t.guid;
    }

    // ── entities: nearby units + gameobjects ────────────────────────────
    std::vector<NBEntityRec> entities;
    std::vector<ObjectGuid> entityGuids; // parallel to entities — true ObjectGuids for action resolution
    size_t nPlayerEntities = 0;
    entities.reserve(NB_MAX_ENTITIES + 16);
    if (bot->IsInWorld())
    {
        float range = NB_SENSE_RANGE;

        std::list<Unit*> units;
        AnyUnitInRangeCheck unitCheck(bot, range);
        Acore::UnitListSearcher<AnyUnitInRangeCheck> unitSearcher(bot, units, unitCheck);
        Cell::VisitObjects(bot, unitSearcher, range);

        for (Unit* u : units)
        {
            if (!u || !u->IsAlive())
                continue;
            // Critters are observation noise (unattackable, uninteresting).
            if (u->IsCritter())
                continue;
            // Cap players at the 4 nearest: 400 bots share racial spawn points, so
            // the distance-sorted 64-slot entity list was 84% friendly bot-clutter at
            // dist 0-3 yd — hostile mobs never entered the frame at all. 4 slots keep
            // player awareness without crowding out creatures/gameobjects.
            if (u->IsPlayer() && ++nPlayerEntities > 4)
                continue;
            NBEntityRec rec{};
            rec.guid = u->GetGUID().GetRawValue();
            rec.entry = u->GetEntry();
            rec.type = UnitEntityType(u);
            rec.level = u->GetLevel();
            rec.health = static_cast<float>(u->GetHealth());
            rec.maxHealth = static_cast<float>(u->GetMaxHealth());
            rec.dx = u->GetPositionX() - bot->GetPositionX();
            rec.dy = u->GetPositionY() - bot->GetPositionY();
            rec.dz = u->GetPositionZ() - bot->GetPositionZ();
            rec.distance = bot->GetDistance(u);
            rec.reaction = ComputeReaction(bot, u);
            rec.alive = 1;
            rec.inCombat = u->IsInCombat() ? 1 : 0;
            rec.casting = u->HasUnitState(UNIT_STATE_CASTING) ? 1 : 0;
            rec.npcFlags = u->ToCreature() ? static_cast<uint32_t>(u->GetNpcFlags()) : 0;
            entities.push_back(rec);
            entityGuids.push_back(u->GetGUID());
        }

        std::list<GameObject*> gos;
        AnyGameObjectInRangeCheck goCheck(bot, range);
        Acore::GameObjectListSearcher<AnyGameObjectInRangeCheck> goSearcher(bot, gos, goCheck);
        Cell::VisitObjects(bot, goSearcher, range);

        for (GameObject* go : gos)
        {
            if (!go)
                continue;
            NBEntityRec rec{};
            rec.guid = go->GetGUID().GetRawValue();
            rec.entry = go->GetEntry();
            rec.type = NB_ENTITY_TYPE_GAMEOBJECT;
            // For gameobjects, npcFlags carries the goType (role: CHEST/DOOR/QUESTGIVER/…)
            rec.npcFlags = static_cast<uint32_t>(go->GetGoType());
            rec.dx = go->GetPositionX() - bot->GetPositionX();
            rec.dy = go->GetPositionY() - bot->GetPositionY();
            rec.dz = go->GetPositionZ() - bot->GetPositionZ();
            rec.distance = bot->GetDistance(go);
            rec.reaction = NB_REACTION_NEUTRAL;
            rec.alive = 1;
            entities.push_back(rec);
            entityGuids.push_back(go->GetGUID());
        }
    }

    // Sort records and guids together by distance (frame order = nearest first)
    {
        std::vector<size_t> order(entities.size());
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
            { return entities[a].distance < entities[b].distance; });
        std::vector<NBEntityRec> sortedRecs(order.size());
        std::vector<ObjectGuid> sortedGuids(order.size());
        for (size_t i = 0; i < order.size(); ++i)
        {
            sortedRecs[i] = entities[order[i]];
            sortedGuids[i] = entityGuids[order[i]];
        }
        entities.swap(sortedRecs);
        entityGuids.swap(sortedGuids);
    }

    size_t nEntities = std::min<size_t>(entities.size(), NB_MAX_ENTITIES);
    // Optionally return guids in frame order so TARGET_ENTITY_i (action) resolves to
    // the same entity the policy observed at this step.
    if (entityCountOut)
        *entityCountOut = nEntities;
    for (size_t i = 0; i < nEntities; ++i)
    {
        frame.entities[i] = entities[i];
        if (entityGuidsOut)
            entityGuidsOut[i] = entityGuids[i];
    }
    frame.counts.nEntities = static_cast<uint16_t>(nEntities);

    // ── spells ──────────────────────────────────────────────────────────
    {
        PlayerSpellMap const& spellMap = bot->GetSpellMap();
        // PlayerSpellMap is std::unordered_map, so iteration order is arbitrary — a
        // policy could never learn CAST_SPELL_i semantics. Sort by spellId so the
        // frame order (and hence CAST_SPELL_0..7) is deterministic per bot.
        std::vector<NBSpellRec> spells;
        spells.reserve(NB_MAX_SPELLS);
        for (auto const& [spellId, state] : spellMap)
        {
            if (state->State == PLAYERSPELL_REMOVED)
                continue;
            SpellInfo const* info = sSpellMgr->GetSpellInfo(spellId);
            if (!info)
                continue;
            // Castables only — a real client's action bar doesn't offer passives either.
            // Keeps CAST_SPELL_i indices useful for exploration.
            if (info->IsPassive())
                continue;
            NBSpellRec rec{};
            rec.spellId = spellId;
            rec.cooldownMs = bot->GetSpellCooldownDelay(spellId);
            rec.ready = !bot->HasSpellCooldown(spellId) ? 1 : 0;
            rec.cost = info->ManaCost;
            rec.range = info->GetMaxRange(true);
            rec.minRange = info->GetMinRange(true);
            rec.castTimeMs = info->CastTimeEntry ? static_cast<float>(info->CastTimeEntry->CastTime) : 0.0f;
            spells.push_back(rec);
        }
        std::sort(spells.begin(), spells.end(),
            [](NBSpellRec const& a, NBSpellRec const& b) { return a.spellId < b.spellId; });
        size_t n = std::min<size_t>(spells.size(), NB_MAX_SPELLS);
        for (size_t i = 0; i < n; ++i)
            frame.spells[i] = spells[i];
        frame.counts.nSpells = static_cast<uint16_t>(n);
    }

    // ── quests ──────────────────────────────────────────────────────────
    {
        size_t n = 0;
        for (uint32 slot = 0; slot < 25 && n < NB_MAX_QUESTS; ++slot)
        {
            uint32 qid = bot->GetQuestSlotQuestId(slot);
            if (!qid)
                continue;
            NBQuestRec& rec = frame.quests[n];
            rec.questId = qid;
            rec.status = static_cast<uint8_t>(bot->GetQuestStatus(qid));
            rec.obj[0] = bot->GetQuestSlotCounter(slot, 0);
            rec.obj[1] = bot->GetQuestSlotCounter(slot, 1);
            rec.obj[2] = bot->GetQuestSlotCounter(slot, 2);
            rec.obj[3] = bot->GetQuestSlotCounter(slot, 3);
            ++n;
        }
        frame.counts.nQuests = static_cast<uint16_t>(n);
    }

    // ── items: nearby lootable corpses + chest gameobjects ──────────────
    {
        std::vector<NBItemRec> items;
        items.reserve(NB_MAX_ITEMS + 8);
        if (bot->IsInWorld())
        {
            float range = 30.0f;

            std::list<Unit*> corpses;
            AnyUnitInRangeCheck corpseCheck(bot, range);
            Acore::UnitListSearcher<AnyUnitInRangeCheck> corpseSearcher(bot, corpses, corpseCheck);
            Cell::VisitObjects(bot, corpseSearcher, range);

            for (Unit* u : corpses)
            {
                if (!u || !u->IsCreature() || u->IsAlive())
                    continue;
                NBItemRec rec{};
                rec.guid = u->GetGUID().GetRawValue();
                rec.entry = u->GetEntry();
                rec.quality = 0;
                rec.distance = bot->GetDistance(u);
                items.push_back(rec);
            }

            std::list<GameObject*> chests;
            AnyGameObjectInRangeCheck chestCheck(bot, range);
            Acore::GameObjectListSearcher<AnyGameObjectInRangeCheck> chestSearcher(bot, chests, chestCheck);
            Cell::VisitObjects(bot, chestSearcher, range);

            for (GameObject* go : chests)
            {
                if (!go || go->GetGoType() != GAMEOBJECT_TYPE_CHEST)
                    continue;
                NBItemRec rec{};
                rec.guid = go->GetGUID().GetRawValue();
                rec.entry = go->GetEntry();
                rec.quality = 0;
                rec.distance = bot->GetDistance(go);
                items.push_back(rec);
            }
        }

        std::sort(items.begin(), items.end(),
            [](NBItemRec const& a, NBItemRec const& b) { return a.distance < b.distance; });

        size_t nItems = std::min<size_t>(items.size(), NB_MAX_ITEMS);
        for (size_t i = 0; i < nItems; ++i)
            frame.items[i] = items[i];
        frame.counts.nItems = static_cast<uint16_t>(nItems);
    }
}

void NeuralBotInstance::BuildFrame(NeuralBotFrame& frame)
{
    BuildFrameFor(_player, frame, _frameEntityGuids, &_frameEntityCount);
}

float NeuralBotInstance::ComputeReward(NeuralBotReward& out)
{
    out = NeuralBotReward{};
    Player* bot = _player;
    if (!bot) return 0.0f;

    // ── Native reward terms ────────────────────────────────────────────────
    // These are the ONLY signals summed into `total` (see DESIGN.md). Everything
    // below this section is computed for logging/diagnostics only — no shaping.

    // XP — the core leveling signal. Kill XP, quest-turn-in XP, and exploration
    // XP all flow through PLAYER_XP. On a level-up the counter resets with
    // carry-over, so reconstruct the true gain across the threshold (otherwise a
    // level-up would read as a large negative delta).
    float curXp = static_cast<float>(bot->GetUInt32Value(PLAYER_XP));
    float curNextXp = static_cast<float>(bot->GetUInt32Value(PLAYER_NEXT_LEVEL_XP));
    uint32 curLevel = bot->GetLevel();

    float xpGained;
    if (curLevel > _prevLevel && _prevLevel > 0)
        xpGained = (_prevNextLevelXp - _prevXp) + curXp;
    else
        xpGained = curXp - _prevXp;
    out.xpDelta = xpGained / 100.0f;

    // Level — sparse milestone bonus; leveling is the explicit objective.
    float levelReward = (curLevel > _prevLevel && _prevLevel > 0) ? 1.0f : 0.0f;

    _prevXp = curXp;
    _prevNextLevelXp = curNextXp;
    _prevLevel = curLevel;

    // Death — sparse penalty. Death is genuinely costly (corpse run + repair).
    if (_diedThisStep)
    {
        out.deathPenalty = 10.0f;
        _diedThisStep = false;
    }

    // Gold — money earned from loot, quests, and vendoring (copper → gold).
    float curMoney = static_cast<float>(bot->GetMoney());
    out.lootReward = (curMoney - _prevMoney) / 10000.0f;
    _prevMoney = curMoney;

    // ── Additional tracked terms ──────────────────────────────────────────
    // Quest completion and spell-learned feed the native total above; the rest are
    // computed for logging/diagnostics only (no shaping, per DESIGN.md).

    float killReward = _killCount * 5.0f;
    out.killReward = killReward;
    _killCount = 0.0f;

    float curHp = static_cast<float>(bot->GetHealth());
    float hpDelta = _prevHealth - curHp;
    float damagePenalty = 0.0f;
    if (hpDelta > 0)
        damagePenalty = hpDelta / static_cast<float>(bot->GetMaxHealth()) * 0.5f;
    out.damageTaken = damagePenalty;
    _prevHealth = curHp;

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
                questAcceptedReward += 5.0f;

        for (auto const& [qid, prevStatus] : _prevQuestStatus)
        {
            auto it = curStatus.find(qid);
            if (it != curStatus.end())
            {
                uint8 cur = it->second;
                if (prevStatus == static_cast<uint8>(QUEST_STATUS_COMPLETE) &&
                    cur == static_cast<uint8>(QUEST_STATUS_REWARDED))
                    questCompletedReward += 20.0f;
            }
        }

        for (auto const& [qid, prevCounts] : _prevObjectiveCounts)
        {
            auto it = curCounters.find(qid);
            if (it != curCounters.end())
                for (int i = 0; i < 4; ++i)
                    if (it->second[i] > prevCounts[i])
                        questProgressReward += 0.5f * static_cast<float>(it->second[i] - prevCounts[i]);
        }

        if (_cachedNearestQGDist > 0.0f)
        {
            if (_cachedNearestQGDist < 5.0f)
                questProximityReward = 0.05f * (1.0f - _cachedNearestQGDist / 5.0f);
            else if (_cachedNearestQGDist < 10.0f)
                questProximityReward = 0.025f;
            else if (_cachedNearestQGDist < 20.0f)
                questProximityReward = 0.01f;
            else
                questProximityReward = 0.005f;

            if (_prevQGDist > 0.0f && _cachedNearestQGDist < _prevQGDist)
            {
                float pct = (_prevQGDist - _cachedNearestQGDist) / _prevQGDist;
                questProximityReward += std::min(pct * 0.005f, 0.01f);
            }
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

    float enemyProximityReward = 0.0f;
    if (_cachedNearestEnemyDist > 0.0f)
    {
        if (_cachedNearestEnemyDist < 5.0f)
            enemyProximityReward = 0.5f * (1.0f - _cachedNearestEnemyDist / 5.0f);
        else if (_cachedNearestEnemyDist < 10.0f)
            enemyProximityReward = 0.25f;
        else if (_cachedNearestEnemyDist < 20.0f)
            enemyProximityReward = 0.1f;
        else
            enemyProximityReward = 0.05f;

        if (_prevEnemyDist > 0.0f && _cachedNearestEnemyDist < _prevEnemyDist)
        {
            float pct = (_prevEnemyDist - _cachedNearestEnemyDist) / _prevEnemyDist;
            enemyProximityReward += std::min(pct * 0.05f, 0.1f);
        }
        _prevEnemyDist = _cachedNearestEnemyDist;
    }
    out.enemyProximity = enemyProximityReward;

    float targetAcquiredReward = 0.0f;
    float damageDealt = 0.0f;
    {
        Unit* target = bot->GetSelectedUnit();
        ObjectGuid curTarget = target ? target->GetGUID() : ObjectGuid::Empty;
        bool validEnemy = target && target->ToCreature() && !target->IsFriendlyTo(bot);
        if (validEnemy && !curTarget.IsEmpty())
        {
            if (curTarget != _prevTargetGuid)
                targetAcquiredReward = 0.5f;

            // Damage DEALT = selected target's hp drop since last step. Dense, positive,
            // and un-gameable (hitting requires engaging the enemy, which invites damage
            // taken + death risk).
            float tHp = static_cast<float>(target->GetHealth());
            if (curTarget == _prevTargetGuid && _prevTargetHealth > 0.0f)
            {
                float dealt = _prevTargetHealth - tHp;
                if (dealt > 0.0f)
                    damageDealt = dealt / static_cast<float>(target->GetMaxHealth());
            }
            _prevTargetHealth = tHp;
        }
        else
        {
            _prevTargetHealth = 0.0f;
        }
        _prevTargetGuid = curTarget;
    }
    out.targetAcquired = targetAcquiredReward;
    out.damageDealt = damageDealt;

    float trainerProximityReward = 0.0f;
    if (_cachedNearestTrainerDist > 0.0f)
    {
        if (_cachedNearestTrainerDist < 5.0f)
            trainerProximityReward = 0.1f * (1.0f - _cachedNearestTrainerDist / 5.0f);
        else if (_cachedNearestTrainerDist < 10.0f)
            trainerProximityReward = 0.05f;
        else if (_cachedNearestTrainerDist < 20.0f)
            trainerProximityReward = 0.02f;
        else
            trainerProximityReward = 0.005f;

        if (_prevTrainerDist > 0.0f && _cachedNearestTrainerDist < _prevTrainerDist)
            trainerProximityReward += 0.002f;
        _prevTrainerDist = _cachedNearestTrainerDist;
    }
    out.trainerProximity = trainerProximityReward;

    out.spellLearned = _spellsLearnedThisEpisode * 10.0f;
    _spellsLearnedThisEpisode = 0;

    out.timePenalty = -0.001f;

    // ── Combat-bootstrap shaping ───────────────────────────────────────────
    // Two correctly-credited terms. The earlier proximity potential was removed:
    // the ATTACK_START approach is a *sticky* MovePoint, so distance-decrease reward
    // was credited to whatever action the bot took during the move (usually NOOP),
    // reinforcing NOOP instead of the initiating action.
    //  1. Target potential (PBRS): +1.0 acquire / -1.0 lose a valid enemy target.
    //     Credited to the action that caused the acquisition (target or attack).
    //  2. Attack-engagement reward: +0.3 the first time each enemy is engaged via
    //     ACTION_ATTACK_START. Credited directly to the attack action, bootstrapping
    //     the one-action engage (auto-target + approach + swing).
    float potential = 0.0f;
    {
        Unit* sel = bot->GetSelectedUnit();
        if (sel && sel->ToCreature() && !sel->IsFriendlyTo(bot) && sel->IsAlive())
            potential += 1.0f;
    }
    float shaping = 0.999f * potential - _prevPotential;
    _prevPotential = potential;

    float attackEngagedReward = 0.0f;
    if (_didAttackThisStep)
    {
        Unit* tgt = bot->GetSelectedUnit();
        // Reward only a *close* first engagement. Rewarding every far-target attack
        // taught the policy to hop between distant mobs (20 unique targets/ep, 0 kills)
        // instead of closing to melee and dealing damage.
        if (tgt && tgt->IsAlive() && !tgt->IsFriendlyTo(bot) && bot->GetDistance(tgt) < 8.0f)
        {
            uint64 g = tgt->GetGUID().GetRawValue();
            if (_engagedGuids.insert(g).second) // once per enemy per episode (set, not
                attackEngagedReward = 0.1f;      // last-GUID: A→B→A oscillation farmed 0.3 each switch). 0.1: bootstrap only — kills (5.0) must dominate
        }
        _didAttackThisStep = false;
    }

    // Dense reward (v0.8.1): the native milestones alone are too sparse (the
    // world-model reward head stayed at 0.0 for 20M steps), but the first dense
    // attempt (proximity + target-acquired) was immediately gamed (scores 1000+ by
    // target-switching). Keep ONLY non-gameable dense terms: a constant time penalty
    // (every step non-zero), damage dealt (dense positive, requires engaging the
    // enemy), and damage taken (dense negative, teaches efficient combat).
    // killReward (5.0/kill) is IN the total: without it, engagement (0.3 x ~7 mobs)
    // outpaid killing (~0.5 xp-equivalent) — the policy optimized mob-hopping over
    // completion (live measurement: 0.2 kills/ep, 19% of mobs chipped, 5% below half).
    return out.xpDelta + out.lootReward + levelReward + out.questAccepted + out.questCompleted + out.spellLearned
         + out.questProgress
         + out.killReward
         + out.damageDealt
         - out.deathPenalty - out.damageTaken + out.timePenalty + shaping + attackEngagedReward;
}

void NeuralBotInstance::ResetRewardTracking()
{
    if (!_player) return;
    _prevXp = static_cast<float>(_player->GetUInt32Value(PLAYER_XP));
    _prevNextLevelXp = static_cast<float>(_player->GetUInt32Value(PLAYER_NEXT_LEVEL_XP));
    _prevLevel = _player->GetLevel();
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
    _cachedNearestEnemyDist = 0.0f;
    _prevEnemyDist = 0.0f;
    _prevPotential = 0.0f;
    _didAttackThisStep = false;
    _engagedGuids.clear();
    _moveTargetGuid.Clear();
    _lastMovePathMs = 0;
    _prevTargetGuid = ObjectGuid::Empty;
    _questAutoCompleted = 0;
    _stepsWithoutReward = 0;
    _lastKilledGuid.Clear();
    _prevMoney = _player ? static_cast<float>(_player->GetMoney()) : 0.0f;
    _killsThisEpisode = 0;
    _prevTrainerDist = 0.0f;
    _prevTrainerGuid.Clear();
    _spellsLearnedThisEpisode = 0;
}

Unit* NeuralBotInstance::ResolveFrameEntity(size_t index)
{
    if (index >= _frameEntityCount)
        return nullptr;
    return ObjectAccessor::GetUnit(*_player, _frameEntityGuids[index]);
}

GameObject* NeuralBotInstance::ResolveFrameEntityGO(size_t index)
{
    if (index >= _frameEntityCount)
        return nullptr;
    return _player->GetMap()->GetGameObject(_frameEntityGuids[index]);
}

uint32 NeuralBotInstance::GetFrameSpellId(size_t index)
{
    // Mirror BuildFrame's spells[] enumeration exactly (PlayerSpellMap order, skipping
    // removed/unknown) so CAST_SPELL_i addresses the same spell the policy observed.
    Player* bot = _player;
    if (!bot)
        return 0;
    PlayerSpellMap const& spellMap = bot->GetSpellMap();
    size_t n = 0;
    for (auto const& [spellId, state] : spellMap)
    {
        if (n >= NB_MAX_SPELLS)
            break;
        if (state->State == PLAYERSPELL_REMOVED)
            continue;
        if (!sSpellMgr->GetSpellInfo(spellId))
            continue;
        if (sSpellMgr->GetSpellInfo(spellId)->IsPassive()) // mirror BuildFrame's castables-only filter
            continue;
        if (n == index)
            return spellId;
        ++n;
    }
    return 0;
}

namespace
{
    Unit* FindNearestMatchingUnit(Player* bot, float range, bool hostiles, bool wantDead)
    {
        std::list<Unit*> units;
        if (hostiles)
        {
            Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, range);
            Acore::UnitListSearcher<decltype(check)> searcher(bot, units, check);
            Cell::VisitObjects(bot, searcher, range);
        }
        else
        {
            Acore::AnyFriendlyUnitInObjectRangeCheck check(bot, bot, range);
            Acore::UnitListSearcher<decltype(check)> searcher(bot, units, check);
            Cell::VisitObjects(bot, searcher, range);
        }

        Unit* best = nullptr;
        float bestDist = range + 1.0f;
        for (Unit* u : units)
        {
            if (!u || u == bot)
                continue;
            // Players are never valid combat targets: unflagged enemy players silently
            // reject ATTACKSWING, which re-created the friendly-lock (bots parked next
            // to opposite-faction bots, 0 kills, once zone mobs were depleted).
            if (hostiles && u->IsPlayer())
                continue;
            if (u->IsAlive() == wantDead)
                continue;
            if (!hostiles && !u->IsAlive()) // friendly scan: alive only
                continue;
            float d = bot->GetDistance(u);
            if (d < bestDist)
            {
                bestDist = d;
                best = u;
            }
        }
        return best;
    }

    GameObject* FindNearestChestGO(Player* bot, float range)
    {
        std::list<GameObject*> gos;
        AnyGameObjectInRangeCheck goCheck(bot, range);
        Acore::GameObjectListSearcher<AnyGameObjectInRangeCheck> goSearcher(bot, gos, goCheck);
        Cell::VisitObjects(bot, goSearcher, range);

        GameObject* best = nullptr;
        float bestDist = range + 1.0f;
        for (GameObject* go : gos)
        {
            if (!go || go->GetGoType() != GAMEOBJECT_TYPE_CHEST)
                continue;
            float d = bot->GetDistance(go);
            if (d < bestDist)
            {
                bestDist = d;
                best = go;
            }
        }
        return best;
    }
}

void NeuralBotInstance::ExecuteAction(uint32 action)
{
    Player* bot = _player;
    if (!bot || !bot->IsAlive())
        return;

    LOG_DEBUG("module.neuralbot", "Instance '{}' Action: {}", GetName(), action);

    switch (action)
    {
    case ACTION_NOOP:
        break;

    case ACTION_MOVE_TO_TARGET:
    {
        // Point navigation toward the selected unit. MoveChase is a creature-only
        // generator (no-op on players); playerbots-style pathed MovePoint keeps the bot
        // walking across ticks until arrival. Fallback chain keeps a random policy
        // mobile: nearest hostile, then nearest chest.
        Unit* target = bot->GetSelectedUnit();
        if (!target || !target->IsInWorld())
            target = FindNearestMatchingUnit(bot, NB_SENSE_RANGE, true, false);
        if (target)
        {
            MotionMaster* mm = bot->GetMotionMaster();
            mm->Clear();
            mm->MovePoint(0, target->GetPositionX(), target->GetPositionY(), target->GetPositionZ(),
                FORCED_MOVEMENT_NONE, 0.0f, 0.0f, /*generatePath=*/true, /*forceDestination=*/false);
            break;
        }
        if (GameObject* go = FindNearestChestGO(bot, 40.0f))
            bot->GetMotionMaster()->MovePoint(0, go->GetPositionX(), go->GetPositionY(), go->GetPositionZ(),
                FORCED_MOVEMENT_NONE, 0.0f, 0.0f, true, false);
        break;
    }

    case ACTION_STOP_MOVE:
        bot->GetMotionMaster()->Clear();
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
        bot->GetMotionMaster()->MovePoint(0, x, y, z, FORCED_MOVEMENT_NONE, 0.0f, 0.0f, true, false);
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
        bot->GetMotionMaster()->MovePoint(0, x, y, z, FORCED_MOVEMENT_NONE, 0.0f, 0.0f, true, false);
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

    case ACTION_TARGET_NEAREST_ENEMY:
    {
        if (Unit* nearest = FindNearestMatchingUnit(bot, NB_SENSE_RANGE, true, false))
            InjectCMSG(CMSG_SET_SELECTION, [nearest](WorldPacket& pkt) { pkt << nearest->GetGUID(); });
        break;
    }
    case ACTION_TARGET_NEAREST_FRIENDLY:
    {
        if (Unit* nearest = FindNearestMatchingUnit(bot, NB_SENSE_RANGE, false, false))
            InjectCMSG(CMSG_SET_SELECTION, [nearest](WorldPacket& pkt) { pkt << nearest->GetGUID(); });
        break;
    }
    case ACTION_TARGET_NEAREST_CORPSE:
    {
        if (Unit* nearest = FindNearestMatchingUnit(bot, NB_SENSE_RANGE, true, true))
            InjectCMSG(CMSG_SET_SELECTION, [nearest](WorldPacket& pkt) { pkt << nearest->GetGUID(); });
        break;
    }

    default:
        break;
    }

    // TARGET_ENTITY_0 .. TARGET_ENTITY_LAST — select the i-th nearest frame entity
    if (action >= ACTION_TARGET_ENTITY_0 && action <= ACTION_TARGET_ENTITY_LAST)
    {
        if (Unit* u = ResolveFrameEntity(action - ACTION_TARGET_ENTITY_0))
            InjectCMSG(CMSG_SET_SELECTION, [u](WorldPacket& pkt) { pkt << u->GetGUID(); });
        return;
    }

    if (action == ACTION_ATTACK_START)
    {
        Unit* target = bot->GetSelectedUnit();
        // Validate the selection: ATTACKSWING on a friendly/dead unit is rejected
        // server-side SILENTLY — the bot then parks at 0 yd "attacking" it forever
        // (observed live: 63% of bots locked on friendly targets, 5% combat, 0 kills
        // while spamming ATTACK_START ~700x/episode). Retarget to a real hostile.
        if (!target || !target->IsAlive() || target->IsFriendlyTo(bot))
            target = FindNearestMatchingUnit(bot, NB_SENSE_RANGE, true, false); // interim auto-service (ROADMAP §3)
        if (target)
        {
            InjectCMSG(CMSG_SET_SELECTION, [target](WorldPacket& pkt) { pkt << target->GetGUID(); });
            // Interim scaffold: also close distance when out of melee reach. The level-1
            // population sees mobs (50%) but rarely sustains MOVE→ATTACK long enough to
            // engage (1% combat); the leveled cohort that enters the loop keeps fighting
            // (64% combat @ lvl5) — collapsing approach+attack into one action gives the
            // stuck mass a single-action path into the loop. Remove with §3 cleanup.
            if (bot->GetDistance(target) > 4.5f && target->IsAlive())
            {
                // Throttle re-pathing: re-issue MovePoint only on a NEW target or at
                // most once per second. Re-pathing every step (policy picks
                // ATTACK_START ~1.5x/step) kept resetting the approach so bots never
                // arrived in melee; never re-pathing misses mobs that move between
                // ticks. 1s cadence tracks movement while letting the path run.
                uint32 now = getMSTime();
                if (_moveTargetGuid != target->GetGUID() || now - _lastMovePathMs >= 1000)
                {
                    _moveTargetGuid = target->GetGUID();
                    _lastMovePathMs = now;
                    MotionMaster* mm = bot->GetMotionMaster();
                    mm->Clear();
                    mm->MovePoint(0, target->GetPositionX(), target->GetPositionY(), target->GetPositionZ(),
                        FORCED_MOVEMENT_NONE, 0.0f, 0.0f, /*generatePath=*/true, /*forceDestination=*/false);
                }
            }
            InjectCMSG(CMSG_ATTACKSWING, [target](WorldPacket& pkt) { pkt << target->GetGUID(); });
            _didAttackThisStep = true; // reward the *initiating* action (see ComputeReward)
        }
        return;
    }
    if (action == ACTION_ATTACK_STOP)
    {
        InjectCMSG(CMSG_ATTACKSTOP, [](WorldPacket& pkt) { });
        return;
    }

    // CAST_SPELL_0 .. CAST_SPELL_LAST — i-th spellbook entry (frame spells[] order)
    if (action >= ACTION_CAST_SPELL_0 && action <= ACTION_CAST_SPELL_LAST)
    {
        uint32 spellId = GetFrameSpellId(action - ACTION_CAST_SPELL_0);
        if (spellId == 0)
            return;
        Unit* target = bot->GetSelectedUnit();
        if (!target)
            target = FindNearestMatchingUnit(bot, NB_SENSE_RANGE, true, false); // §3: remove this auto-service later
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
        return;
    }

    if (action == ACTION_INTERACT_TARGET)
    {
        DoInteractWithTarget();
        return;
    }

    if (action == ACTION_COMPLETE_QUEST)
    {
        ExecuteActionLegacyQuestTurnIn();
        return;
    }

    if (action == ACTION_LOOT)
    {
        ExecuteActionLegacyLoot();
        return;
    }
}

void NeuralBotInstance::DoInteractWithTarget()
{
    Player* bot = _player;
    if (!bot || !bot->IsAlive() || !bot->IsInWorld())
        return;

    Unit* target = bot->GetSelectedUnit();

    // No unit selected (or a dead one nearby): try nearest chest gameobject in reach.
    if (!target)
    {
        if (GameObject* go = FindNearestChestGO(bot, 5.5f))
        {
            ObjectGuid guid = go->GetGUID();
            InjectCMSG(CMSG_GAMEOBJ_USE, [guid](WorldPacket& pkt) { pkt << guid; });
            InjectCMSG(CMSG_GAMEOBJ_USE, [guid](WorldPacket& pkt) { pkt << guid; }); // open + loot open
        }
        return;
    }

    // The client enforces ~5 yards for interactions; teaching approach → interact is
    // exactly what fixes the historical "found trainer, never learned" failure.
    float dist = bot->GetDistance(target);
    if (dist > 5.5f)
    {
        LOG_DEBUG("module.neuralbot", "'{}' INTERACT too far ({:.1f} yd, entry {})", GetName(), dist, target->GetEntry());
        return;
    }

    Creature* creature = target->ToCreature();
    if (!creature)
        return;

    uint32 npcFlags = creature->GetNpcFlags();
    ObjectGuid guid = creature->GetGUID();

    if (npcFlags & UNIT_NPC_FLAG_QUESTGIVER)
    {
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
        return;
    }

    if (npcFlags & UNIT_NPC_FLAG_TRAINER)
    {
        Trainer::Trainer const* trainer = sObjectMgr->GetTrainer(creature->GetEntry());
        if (!trainer || !trainer->IsTrainerValidForPlayer(bot))
            return;
        for (auto const& spell : trainer->GetSpells())
        {
            if (trainer->CanTeachSpell(bot, &spell) && spell.MoneyCost <= bot->GetMoney())
            {
                InjectCMSG(CMSG_TRAINER_BUY_SPELL, [guid, spellId = spell.SpellId](WorldPacket& pkt) {
                    pkt << guid;
                    pkt << uint32(spellId);
                });
                break;
            }
        }
        return;
    }

    if (npcFlags & UNIT_NPC_FLAG_VENDOR)
        InjectCMSG(CMSG_LIST_INVENTORY, [guid](WorldPacket& pkt) { pkt << guid << uint32(0) << uint8(0) << uint8(0) << uint8(0); });
}

void NeuralBotInstance::ExecuteActionLegacyQuestTurnIn()
{
    Player* bot = _player;
    if (!bot || !bot->IsInWorld())
        return;
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
                    if (quest->GetRewChoiceItemsCount() > 0)
                    {
                        uint32 choice = ChooseQuestReward(quest);
                        InjectCMSG(CMSG_QUESTGIVER_CHOOSE_REWARD, [guid, qid, choice](WorldPacket& pkt) { pkt << guid << uint32(qid) << choice; });
                    }
                    completed = true;
                    break;
                }
            }
            if (completed) break;
        }
    }
}

uint32 NeuralBotInstance::ChooseQuestReward(Quest const* quest)
{
    Player* bot = _player;
    uint32 count = quest->GetRewChoiceItemsCount();
    if (count <= 1 || !bot)
        return 0;

    // Class/spec stat weights (simplified from mod-playerbots' StatsWeightCalculator).
    float strW = 0.0f, agiW = 0.0f, intW = 0.0f, staW = 0.0f, spiW = 0.0f;
    switch (bot->getClass())
    {
        case CLASS_WARRIOR:      strW = 2.0f; staW = 1.5f; agiW = 0.5f; break;
        case CLASS_PALADIN:      strW = 2.0f; staW = 1.5f; intW = 0.5f; spiW = 0.5f; break;
        case CLASS_HUNTER:       agiW = 2.0f; staW = 1.0f; break;
        case CLASS_ROGUE:        agiW = 2.0f; staW = 1.0f; break;
        case CLASS_PRIEST:       intW = 2.0f; spiW = 1.5f; staW = 1.0f; break;
        case CLASS_SHAMAN:       intW = 1.5f; agiW = 1.0f; strW = 1.0f; staW = 1.0f; break;
        case CLASS_MAGE:         intW = 2.5f; staW = 1.0f; break;
        case CLASS_WARLOCK:      intW = 2.0f; staW = 1.5f; break;
        case CLASS_DRUID:        agiW = 1.5f; strW = 1.0f; intW = 1.0f; staW = 1.0f; break;
        case CLASS_DEATH_KNIGHT: strW = 2.0f; staW = 1.5f; break;
        default: break;
    }

    auto statValue = [](ItemTemplate const* proto, uint32 statType) -> float {
        for (uint32 s = 0; s < proto->StatsCount && s < MAX_ITEM_PROTO_STATS; ++s)
            if (proto->ItemStat[s].ItemStatType == statType)
                return static_cast<float>(proto->ItemStat[s].ItemStatValue);
        return 0.0f;
    };

    float bestScore = -1e9f;
    uint32 bestIndex = 0;
    for (uint32 i = 0; i < count; ++i)
    {
        ItemTemplate const* proto = sObjectMgr->GetItemTemplate(quest->RewardChoiceItemId[i]);
        if (!proto)
            continue;

        float score;
        if (bot->CanUseItem(proto) != EQUIP_ERR_OK)
        {
            // Unusable for this class — fall back to vendor gold value.
            score = static_cast<float>(proto->SellPrice);
        }
        else
        {
            float statScore = strW * statValue(proto, ITEM_MOD_STRENGTH)
                            + agiW * statValue(proto, ITEM_MOD_AGILITY)
                            + intW * statValue(proto, ITEM_MOD_INTELLECT)
                            + staW * statValue(proto, ITEM_MOD_STAMINA)
                            + spiW * statValue(proto, ITEM_MOD_SPIRIT);
            float baseScore = (proto->Class == ITEM_CLASS_ARMOR)
                            ? static_cast<float>(proto->Armor) * 0.25f
                            : (proto->Damage[0].DamageMin + proto->Damage[0].DamageMax) * 2.0f;
            score = statScore + baseScore;
        }
        if (score > bestScore)
        {
            bestScore = score;
            bestIndex = i;
        }
    }
    return bestIndex;
}

void NeuralBotInstance::ExecuteActionLegacyLoot()
{
    Player* bot = _player;
    if (!bot)
        return;
    Creature* lootTarget = nullptr;
    if (!_lastKilledGuid.IsEmpty())
    {
        if (Creature* c = bot->GetMap()->GetCreature(_lastKilledGuid))
            if (!c->IsAlive() && bot->GetDistance(c) < 20.0f)
                lootTarget = c;
    }
    if (!lootTarget)
    {
        float bestDist = 15.0f;
        std::list<Unit*> nearby;
        Acore::AnyUnfriendlyUnitInObjectRangeCheck check(bot, bot, bestDist);
        Acore::UnitListSearcher<decltype(check)> searcher(bot, nearby, check);
        Cell::VisitObjects(bot, searcher, bestDist);
        for (Unit* u : nearby)
        {
            if (u->isDead() && u->IsCreature())
            {
                Creature* c = u->ToCreature();
                if (c && bot->GetDistance(c) < bestDist)
                {
                    bestDist = bot->GetDistance(c);
                    lootTarget = c;
                }
            }
        }
    }
    if (!lootTarget) return;

    ObjectGuid guid = lootTarget->GetGUID();
    InjectCMSG(CMSG_LOOT, [guid](WorldPacket& pkt) { pkt << guid; });
    InjectCMSG(CMSG_LOOT_MONEY, [](WorldPacket& pkt) { });
    for (uint8 slot = 0; slot < 16; ++slot)
        InjectCMSG(CMSG_AUTOSTORE_LOOT_ITEM, [slot](WorldPacket& pkt) { pkt << slot; });
    InjectCMSG(CMSG_LOOT_RELEASE, [guid](WorldPacket& pkt) { pkt << guid; });
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

    UpdateIdleTracking(result.reward.total);

    result.done = ShouldTerminate(result.info);
    if (result.done)
    {
        static uint32 doneLogCounter = 0;
        if (++doneLogCounter % 200 == 1)
            LOG_INFO("module.neuralbot.debug", "Step done: '{}' reason={} reward={:.4f} stepsWo={} stepCount={}",
                GetName(), result.info, result.reward.total, _stepsWithoutReward, _stepCount);
    }
    return result;
}

void NeuralBotInstance::UpdateIdleTracking(float rewardTotal)
{
    if (rewardTotal > 0.001f)
        _stepsWithoutReward = 0;
    else
        _stepsWithoutReward++;
}

bool NeuralBotInstance::ShouldTerminate(std::string& info)
{
    bool timedOut = _stepCount >= _maxSteps;
    // ~32 shm steps/s per bot: 1500 steps ≈ 47 s without reward. Mobs sit 40+ yd from
    // the spawn clumps (~190 walk steps); the old 200-step budget terminated episodes
    // mid-approach, so combat almost never started (0.2% of episodes ever saw XP).
    bool idle = _stepsWithoutReward >= 1500;
    if (timedOut || _diedThisStep || (_player && !_player->IsAlive()) || idle)
    {
        info = _diedThisStep ? "died" : (idle ? "idle" : "max_steps");
        return true;
    }
    return false;
}

void NeuralBotInstance::WriteFrameReward(NBStateReward& out, NeuralBotReward const& r)
{
    out.total = r.total;
    out.components[0]  = r.xpDelta;
    out.components[1]  = r.damageTaken;
    out.components[2]  = r.killReward;
    out.components[3]  = r.deathPenalty;
    out.components[4]  = r.lootReward;
    out.components[5]  = r.questAccepted;
    out.components[6]  = r.questCompleted;
    out.components[7]  = r.questProximity;
    out.components[8]  = r.questProgress;
    out.components[9]  = r.enemyProximity;
    out.components[10] = r.targetAcquired;
    out.components[11] = r.spellLearned;
    out.components[12] = r.trainerProximity;
    out.components[13] = r.timePenalty;
}

NeuralBotFrameResult NeuralBotInstance::StepFrame(uint32 action)
{
    NeuralBotFrameResult result;
    if (!_ready || !_player || !_player->IsAlive())
    {
        result.done = true;
        result.info = "Instance not ready or dead";
        std::memset(&result.frame, 0, sizeof(NeuralBotFrame));
        return result;
    }

    auto t0 = std::chrono::steady_clock::now();
    ExecuteAction(action);
    _stepCount++;
    auto t1 = std::chrono::steady_clock::now();

    if (_autoQuestEnabled)
        AutoCompleteQuests();

    NeuralBotReward reward;
    reward.total = ComputeReward(reward);
    auto t2 = std::chrono::steady_clock::now();
    BuildFrame(result.frame);
    WriteFrameReward(result.frame.reward, reward);
    auto t3 = std::chrono::steady_clock::now();

    _accActionMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    _accRewardMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
    _accBuildMs  += std::chrono::duration<double, std::milli>(t3 - t2).count();
    _perfSteps++;

    UpdateIdleTracking(reward.total);

    result.done = ShouldTerminate(result.info);
    return result;
}

void NeuralBotInstance::GetPerfStages(double& actionMs, double& rewardMs, double& buildMs, uint32& steps) const
{
    actionMs = _accActionMs;
    rewardMs = _accRewardMs;
    buildMs = _accBuildMs;
    steps = _perfSteps;
}

void NeuralBotInstance::ClearPerfStages()
{
    _accActionMs = 0.0;
    _accRewardMs = 0.0;
    _accBuildMs = 0.0;
    _perfSteps = 0;
}

NeuralBotFrame NeuralBotInstance::ResetFrame()
{
    if (_player)
        ResetRewardTracking();
    NeuralBotFrame frame;
    std::memset(&frame, 0, sizeof(NeuralBotFrame));
    BuildFrame(frame);
    return frame;
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
