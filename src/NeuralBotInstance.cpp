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
        float range = 60.0f;
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

void NeuralBotInstance::BuildFrame(NeuralBotFrame& frame)
{
    Player* bot = _player;
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
    entities.reserve(NB_MAX_ENTITIES + 16);
    if (bot->IsInWorld())
    {
        float range = 60.0f;

        std::list<Unit*> units;
        AnyUnitInRangeCheck unitCheck(bot, range);
        Acore::UnitListSearcher<AnyUnitInRangeCheck> unitSearcher(bot, units, unitCheck);
        Cell::VisitObjects(bot, unitSearcher, range);

        for (Unit* u : units)
        {
            if (!u || !u->IsAlive())
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
    // Cache guids in frame order so TARGET_ENTITY_i (action) resolves to the same
    // entity the policy observed at this step.
    _frameEntityCount = nEntities;
    for (size_t i = 0; i < nEntities; ++i)
    {
        frame.entities[i] = entities[i];
        _frameEntityGuids[i] = entityGuids[i];
    }
    for (size_t i = 0; i < nEntities; ++i)
        frame.entities[i] = entities[i];
    frame.counts.nEntities = static_cast<uint16_t>(nEntities);

    // ── spells ──────────────────────────────────────────────────────────
    {
        PlayerSpellMap const& spellMap = bot->GetSpellMap();
        size_t n = 0;
        for (auto const& [spellId, state] : spellMap)
        {
            if (n >= NB_MAX_SPELLS)
                break;
            if (state->State == PLAYERSPELL_REMOVED)
                continue;
            SpellInfo const* info = sSpellMgr->GetSpellInfo(spellId);
            if (!info)
                continue;
            // Castables only — a real client's action bar doesn't offer passives either.
            // Keeps CAST_SPELL_i indices useful for exploration.
            if (info->IsPassive())
                continue;
            NBSpellRec& rec = frame.spells[n];
            rec.spellId = spellId;
            rec.cooldownMs = bot->GetSpellCooldownDelay(spellId);
            rec.ready = !bot->HasSpellCooldown(spellId) ? 1 : 0;
            rec.cost = info->ManaCost;
            rec.range = info->GetMaxRange(true);
            rec.minRange = info->GetMinRange(true);
            rec.castTimeMs = info->CastTimeEntry ? static_cast<float>(info->CastTimeEntry->CastTime) : 0.0f;
            ++n;
        }
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

    // ── Shaping terms (diagnostic-only, NOT summed into reward) ────────────

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
    {
        Unit* target = bot->GetSelectedUnit();
        ObjectGuid curTarget = target ? target->GetGUID() : ObjectGuid::Empty;
        if (!curTarget.IsEmpty() && curTarget != _prevTargetGuid)
            if (target->ToCreature() && !target->IsFriendlyTo(bot))
                targetAcquiredReward = 0.5f;
        _prevTargetGuid = curTarget;
    }
    out.targetAcquired = targetAcquiredReward;

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

    out.timePenalty = 0.0f;

    // Native total: XP + gold + level milestone + quest completion − death.
    // Quest-turn-in XP also flows through xpDelta, but the discrete completion
    // event gets its own sparse signal for cleaner long-sequence credit.
    return out.xpDelta + out.lootReward + levelReward + out.questCompleted - out.deathPenalty;
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
            target = FindNearestMatchingUnit(bot, 60.0f, true, false);
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
        if (Unit* nearest = FindNearestMatchingUnit(bot, 60.0f, true, false))
            InjectCMSG(CMSG_SET_SELECTION, [nearest](WorldPacket& pkt) { pkt << nearest->GetGUID(); });
        break;
    }
    case ACTION_TARGET_NEAREST_FRIENDLY:
    {
        if (Unit* nearest = FindNearestMatchingUnit(bot, 60.0f, false, false))
            InjectCMSG(CMSG_SET_SELECTION, [nearest](WorldPacket& pkt) { pkt << nearest->GetGUID(); });
        break;
    }
    case ACTION_TARGET_NEAREST_CORPSE:
    {
        if (Unit* nearest = FindNearestMatchingUnit(bot, 60.0f, true, true))
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
        if (Unit* target = bot->GetSelectedUnit())
            InjectCMSG(CMSG_ATTACKSWING, [target](WorldPacket& pkt) { pkt << target->GetGUID(); });
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
            target = FindNearestMatchingUnit(bot, 60.0f, true, false); // §3: remove this auto-service later
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
                    InjectCMSG(CMSG_QUESTGIVER_CHOOSE_REWARD, [guid, qid](WorldPacket& pkt) { pkt << guid << uint32(qid) << uint32(0); });
                    completed = true;
                    break;
                }
            }
            if (completed) break;
        }
    }
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
    bool idle = _stepsWithoutReward >= 200;
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

    ExecuteAction(action);
    _stepCount++;

    if (_autoQuestEnabled)
        AutoCompleteQuests();

    NeuralBotReward reward;
    reward.total = ComputeReward(reward);
    BuildFrame(result.frame);
    WriteFrameReward(result.frame.reward, reward);

    UpdateIdleTracking(reward.total);

    result.done = ShouldTerminate(result.info);
    return result;
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
