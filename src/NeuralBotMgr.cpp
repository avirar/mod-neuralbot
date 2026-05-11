#include "NeuralBotMgr.h"
#include "NeuralBotFactory.h"
#include "NeuralBotCommon.h"
#include "NeuralBotSharedMem.h"
#include "CharacterCache.h"
#include "Config.h"
#include "DatabaseEnv.h"
#include "Log.h"
#include "ObjectMgr.h"
#include "Player.h"
#include "World.h"
#include "WorldSession.h"
#include "WorldSessionMgr.h"
#include "Opcodes.h"

#include <algorithm>
#include <sstream>
#include <thread>

NeuralBotMgr& NeuralBotMgr::instance()
{
    static NeuralBotMgr inst;
    return inst;
}

void NeuralBotMgr::Initialize()
{
    _enabled = sConfigMgr->GetOption<bool>("NeuralBot.Enable", false);
    if (!_enabled)
        return;

    LOG_INFO("module.neuralbot", "NeuralBot manager initializing...");

    _autoQuest = sConfigMgr->GetOption<bool>("NeuralBot.AutoQuest", true);

    NeuralBotFactory::CreateAccounts();

    auto templates = NeuralBotFactory::GetBotTemplates();
    _botCount = static_cast<uint32_t>(templates.size());
    _botOrder.reserve(_botCount);
    for (auto const& t : templates)
        _botOrder.push_back(t.name);

    if (!sNeuralBotShm.Create(_botCount))
        LOG_ERROR("module.neuralbot", "Shared memory initialization failed — falling back to TCP");
    else
        LOG_INFO("module.neuralbot", "Shared memory ready: {} bots, {:.1f} KB", _botCount, SHM_TOTAL_SIZE / 1024.0f);

    LOG_INFO("module.neuralbot", "NeuralBot manager initialized. Target: {} bots", _botCount);
}

void NeuralBotMgr::Shutdown()
{
    _enabled = false;
    sNeuralBotShm.Destroy();
    for (auto& [name, inst] : _instances)
        delete inst;
    _instances.clear();
    _instancesByGuid.clear();
}

void NeuralBotMgr::SpawnAndLoginBots()
{
    if (!NeuralBotFactory::CreateCharacters())
    {
        LOG_ERROR("module.neuralbot", "Failed to create characters");
        return;
    }

    auto created = NeuralBotFactory::GetCreatedCharacters();

    for (auto const& info : created)
    {
        WorldSession* session = new WorldSession(info.accountId, "", 0x0, nullptr,
            SEC_PLAYER, EXPANSION_WRATH_OF_THE_LICH_KING, time_t(0),
            sWorld->GetDefaultDbcLocale(), 0, false, false, 0, true);

        _pendingLogins.push_back({info.accountId, info.guid, info.name, session});
    }

    LOG_INFO("module.neuralbot", "Queued {} bot(s) for parallel async login", _pendingLogins.size());

    LoginAllBots();
}

void NeuralBotMgr::OnWorldUpdate(uint32 /*diff*/)
{
    if (!_enabled)
        return;

    while (ProcessPendingRequests())
    {
    }

    ProcessSharedMemoryStep();

    for (auto& [name, inst] : _instances)
        inst->ProcessBotPackets();
}

void NeuralBotMgr::ProcessSharedMemoryStep()
{
    if (!sNeuralBotShm.IsCreated())
        return;

    uint8_t actions[SHM_MAX_BOTS];
    if (!sNeuralBotShm.TryReadActions(actions, _botCount))
        return;

    static float obsFlat[SHM_MAX_BOTS * SHM_OBS_PER_BOT];
    static uint8_t dones[SHM_MAX_BOTS];

    for (uint32_t i = 0; i < _botCount; ++i)
    {
        float* botObs = obsFlat + i * SHM_OBS_PER_BOT;

        auto it = _instances.find(_botOrder[i]);
        if (it != _instances.end())
        {
            NeuralBotStepResult result = it->second->Step(static_cast<uint32>(actions[i]));

            result.observation.ToFloatArray(botObs);
            botObs[80] = result.reward.total;

            botObs[81] = result.reward.xpDelta;
            botObs[82] = result.reward.damageTaken;
            botObs[83] = result.reward.killReward;
            botObs[84] = result.reward.deathPenalty;
            botObs[85] = result.reward.lootReward;
            botObs[86] = result.reward.questAccepted;
            botObs[87] = result.reward.questCompleted;
            botObs[88] = result.reward.questProximity;
            botObs[89] = result.reward.questProgress;
            botObs[90] = result.reward.enemyProximity;
            botObs[91] = result.reward.targetAcquired;
            botObs[92] = result.reward.timePenalty;

            dones[i] = result.done ? 1 : 0;

            // When episode ends, reset tracking so next step starts fresh
            if (result.done)
                it->second->ResetRewardTracking();
        }
        else
        {
            std::memset(botObs, 0, SHM_OBS_BYTES);
            dones[i] = 1;
            botObs[80] = -1.0f;
        }
    }

    sNeuralBotShm.WriteObservations(obsFlat, dones, _botCount);
    sNeuralBotShm.SignalObservationsReady();

    // Debug: sample bot 0 reward components every 100 steps
    static uint32 stepSampleCounter = 0;
    if (++stepSampleCounter % 100 == 1)
    {
        float* b0 = obsFlat; // bot 0
        LOG_INFO("module.neuralbot.debug", "SHM step {} bot[0]={} reward={:.4f} kill={:.4f} death={:.4f} xp={:.4f} done={}",
            stepSampleCounter, _botOrder[0], b0[80], b0[83], b0[84], b0[81], dones[0]);
    }
}

void NeuralBotMgr::LoginAllBots()
{
    size_t count = _pendingLogins.size();
    if (count == 0)
    {
        LOG_INFO("module.neuralbot", "No bots to log in");
        return;
    }

    for (auto& pending : _pendingLogins)
    {
        auto holder = std::make_shared<LoginQueryHolder>(pending.accountId, pending.guid);
        if (!holder->Initialize())
        {
            LOG_ERROR("module.neuralbot", "Failed to init LoginQueryHolder for '{}'", pending.name);
            delete pending.session;
            continue;
        }

        WorldSession* session = pending.session;

        sWorld->AddQueryHolderCallback(CharacterDatabase.DelayQueryHolder(holder))
            .AfterComplete([this, session, name = pending.name, guid = pending.guid](SQLQueryHolderBase const& queryHolder)
            {
                try
                {
                    LoginQueryHolder const& lqh = static_cast<LoginQueryHolder const&>(queryHolder);
                    session->HandlePlayerLoginFromDB(lqh);
                }
                catch (std::exception const& e)
                {
                    LOG_ERROR("module.neuralbot", "Login exception for '{}': {}", name, e.what());
                    delete session;
                    return;
                }
                catch (...)
                {
                    LOG_ERROR("module.neuralbot", "Unknown login exception for '{}'", name);
                    delete session;
                    return;
                }

                Player* player = session->GetPlayer();
                if (!player)
                {
                    LOG_ERROR("module.neuralbot", "Login failed for '{}' (no Player)", name);
                    delete session;
                    return;
                }

                NeuralBotInstance* inst = new NeuralBotInstance(player, session);
                _instances[name] = inst;
                _instancesByGuid[player->GetGUID()] = inst;

                if (_autoQuest)
                {
                    inst->SetAutoQuest(true);
                    inst->AutoAcceptQuests();
                }

                LOG_INFO("module.neuralbot", "Bot '{}' logged in (GUID:{} Level:{} Zone:{})",
                    name, player->GetGUID().GetCounter(), uint32(player->GetLevel()), player->GetZoneId());
            });
    }

    _pendingLogins.clear();

    LOG_INFO("module.neuralbot", "Fired {} async login queries in parallel", count);
}

void NeuralBotMgr::OnPlayerLogin(Player* player)
{
    if (!_enabled || !player || !player->GetSession()->IsBot())
        return;

    std::string name = player->GetName();
    if (_instances.find(name) != _instances.end())
        return;

    NeuralBotInstance* inst = new NeuralBotInstance(player, player->GetSession());
    _instances[name] = inst;
    _instancesByGuid[player->GetGUID()] = inst;

    if (_autoQuest)
    {
        inst->SetAutoQuest(true);
        inst->AutoAcceptQuests();
    }

    LOG_INFO("module.neuralbot", "Bot '{}' registered via OnPlayerLogin (GUID:{})",
        name, player->GetGUID().GetCounter());
}

void NeuralBotMgr::OnPlayerJustDied(Player* player)
{
    if (!_enabled || !player) return;
    auto it = _instancesByGuid.find(player->GetGUID());
    if (it != _instancesByGuid.end())
        it->second->OnPlayerJustDied();
}

void NeuralBotMgr::OnPlayerCreatureKill(Player* killer, Creature* killed)
{
    if (!_enabled || !killer) return;

    auto it = _instancesByGuid.find(killer->GetGUID());
    if (it != _instancesByGuid.end())
    {
        it->second->OnPlayerCreatureKill();
        static uint32 killLogCounter = 0;
        if (++killLogCounter % 50 == 1)
            LOG_INFO("module.neuralbot.debug", "KILL hook: '{}' killed '{}' (entry {}) — total kills logged: {}",
                killer->GetName(), killed ? killed->GetName() : "?", killed ? killed->GetEntry() : 0, killLogCounter);
    }
}

void NeuralBotMgr::OnPlayerAfterUpdate(Player* player, uint32 /*diff*/)
{
    if (!_enabled || !player) return;

    auto it = _instancesByGuid.find(player->GetGUID());
    if (it != _instancesByGuid.end())
        it->second->ProcessBotPackets();
}

void NeuralBotMgr::OnPlayerbotPacketSent(Player* player, WorldPacket const* packet)
{
    if (!_enabled || !player || !packet) return;
    auto it = _instancesByGuid.find(player->GetGUID());
    if (it != _instancesByGuid.end())
        RecordOpcodeFor(player, packet->GetOpcode());
}

void NeuralBotMgr::RecordOpcodeFor(Player* player, uint16 opcode)
{
    auto it = _instancesByGuid.find(player->GetGUID());
    if (it != _instancesByGuid.end())
        it->second->RecordOpcode(opcode);
}

NeuralBotStepResult NeuralBotMgr::Step(std::string const& botName, uint32 action)
{
    PendingStep ps;
    ps.botName = botName;
    ps.action = action;
    auto future = ps.promise.get_future();
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _pendingSteps.push_back(std::move(ps));
    }
    return future.get();
}

NeuralBotObservation NeuralBotMgr::Reset(std::string const& botName)
{
    PendingReset pr;
    pr.botName = botName;
    auto future = pr.promise.get_future();
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _pendingResets.push_back(std::move(pr));
    }
    return future.get();
}

bool NeuralBotMgr::ProcessPendingRequests()
{
    std::vector<PendingStep> steps;
    std::vector<PendingReset> resets;
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        if (_pendingSteps.empty() && _pendingResets.empty())
            return false;
        steps = std::move(_pendingSteps);
        resets = std::move(_pendingResets);
        _pendingSteps.clear();
        _pendingResets.clear();
    }

    for (auto& ps : steps)
    {
        NeuralBotStepResult result;
        auto it = _instances.find(ps.botName);
        if (it != _instances.end())
            result = it->second->Step(ps.action);
        else
        {
            result.done = true;
            result.info = "Bot not found: " + ps.botName;
        }
        ps.promise.set_value(std::move(result));
    }

    for (auto& pr : resets)
    {
        NeuralBotObservation obs;
        auto it = _instances.find(pr.botName);
        if (it != _instances.end())
            obs = it->second->Reset();
        pr.promise.set_value(std::move(obs));
    }
    return true;
}

NeuralBotInstance* NeuralBotMgr::GetInstance(std::string const& botName)
{
    auto it = _instances.find(botName);
    return it != _instances.end() ? it->second : nullptr;
}

std::vector<std::string> NeuralBotMgr::GetBotNames() const
{
    return _botOrder;
}
