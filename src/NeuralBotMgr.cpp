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
#include <cstdio>
#include <cstring>

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
    _curriculumStaging = sConfigMgr->GetOption<bool>("NeuralBot.Curriculum", true);
    _levelBandRespawn = sConfigMgr->GetOption<bool>("NeuralBot.LevelBandRespawn", true);
    _cleanupOnStartup = sConfigMgr->GetOption<bool>("NeuralBot.CleanupOnStartup", false);
    _perfLogInterval = sConfigMgr->GetOption<uint32>("NeuralBot.PerfLogInterval", 250);

    // Cleanup must run BEFORE account/character creation so the guid generator and all
    // dependent tables are reset together (accounts + characters + dependents).
    if (_cleanupOnStartup)
        NeuralBotFactory::CleanupBots();

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

    InitBcRecorder();

    LOG_INFO("module.neuralbot", "NeuralBot manager initialized. Target: {} bots, levelBand={}, curriculum={}, autoQuest={}, cleanup={}",
        _botCount, _levelBandRespawn ? 1 : 0, _curriculumStaging ? 1 : 0, _autoQuest ? 1 : 0, _cleanupOnStartup ? 1 : 0);
}

void NeuralBotMgr::Shutdown()
{
    _enabled = false;
    CloseBcRecorder();
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

    // Pipelined protocol backpressure: never overwrite frames the Python reader has
    // not harvested yet (it clears obs_ready within ~1ms). Without this guard a fast
    // C++ side could tear the frame batch Python is copying out.
    if (sNeuralBotShm.ObservationsPending())
        return;

    uint8_t actions[SHM_MAX_BOTS];
    if (!sNeuralBotShm.TryReadActions(actions, _botCount))
        return;

    auto tStep0 = std::chrono::steady_clock::now();

    static uint8_t frames[SHM_MAX_BOTS * SHM_FRAME_BYTES];
    static uint8_t dones[SHM_MAX_BOTS];
    float bot0Reward = 0.0f;

    for (uint32_t i = 0; i < _botCount; ++i)
    {
        uint8_t* botFrame = frames + i * SHM_FRAME_BYTES;

        auto it = _instances.find(_botOrder[i]);
        if (it != _instances.end())
        {
            NeuralBotFrameResult result = it->second->StepFrame(static_cast<uint32>(actions[i]));

            std::memcpy(botFrame, &result.frame, SHM_FRAME_BYTES);
            dones[i] = result.done ? 1 : 0;
            if (i == 0)
                bot0Reward = result.frame.reward.total;

            // When episode ends, reset tracking so next step starts fresh
            if (result.done)
            {
                it->second->ResetRewardTracking();
                it->second->ReviveIfDead();
                if (_levelBandRespawn)
                    it->second->RespawnToLevelBand();
                if (_curriculumStaging)
                    it->second->StageEpisodeStart();
            }
        }
        else
        {
            std::memset(botFrame, 0, SHM_FRAME_BYTES);
            dones[i] = 1;
        }
    }

    sNeuralBotShm.WriteFrames(frames, dones, _botCount);
    sNeuralBotShm.SignalObservationsReady();

    // ── Perf instrumentation: cycle vs step vs wait, plus per-stage breakdown ──
    auto tStep1 = std::chrono::steady_clock::now();
    double stepMs = std::chrono::duration<double, std::milli>(tStep1 - tStep0).count();
    double cycleMs = _perfSamples > 0
        ? std::chrono::duration<double, std::milli>(tStep1 - _lastBatchDone).count()
        : stepMs;
    _lastBatchDone = tStep1;
    _accCycleMs += cycleMs;
    _accStepMs += stepMs;
    _perfSamples++;

    if (_perfSamples % _perfLogInterval == 0)
    {
        double avgCycle = _accCycleMs / _perfLogInterval;
        double avgStep = _accStepMs / _perfLogInterval;
        double actionMs = 0.0, rewardMs = 0.0, buildMs = 0.0;
        uint32 stageSteps = 0;
        for (auto& pair : _instances)
        {
            double a, r, b; uint32 s;
            pair.second->GetPerfStages(a, r, b, s);
            actionMs += a; rewardMs += r; buildMs += b; stageSteps += s;
            pair.second->ClearPerfStages();
        }
        double n = stageSteps > 0 ? static_cast<double>(stageSteps) : 1.0;
        LOG_INFO("module.neuralbot.perf",
            "batches={} cycle={:.2f}ms step={:.2f}ms wait={:.2f}ms (C++ {:.0f}%) "
            "| per-bot: action={:.3f}ms reward={:.3f}ms build={:.3f}ms",
            _perfSamples, avgCycle, avgStep, avgCycle - avgStep,
            avgCycle > 0 ? 100.0 * avgStep / avgCycle : 0.0,
            actionMs / n, rewardMs / n, buildMs / n);
        _accCycleMs = 0.0;
        _accStepMs = 0.0;
    }

    // Debug: sample bot 0 reward every 100 steps
    static uint32 stepSampleCounter = 0;
    if (++stepSampleCounter % 100 == 1)
    {
        LOG_INFO("module.neuralbot.debug", "SHM step {} bot[0]={} reward={:.4f} done={}",
            stepSampleCounter, _botOrder[0], bot0Reward, dones[0]);
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
        it->second->OnPlayerCreatureKill(killed);
        static uint32 killLogCounter = 0;
        if (++killLogCounter % 50 == 1)
            LOG_INFO("module.neuralbot.debug", "KILL hook: '{}' killed '{}' (entry {}) — total kills logged: {}",
                killer->GetName(), killed ? killed->GetName() : "?", killed ? killed->GetEntry() : 0, killLogCounter);
    }
}

void NeuralBotMgr::OnPlayerLearnSpell(Player* player, uint32 spellId)
{
    if (!_enabled || !player) return;
    auto it = _instancesByGuid.find(player->GetGUID());
    if (it != _instancesByGuid.end())
        it->second->OnPlayerLearnSpell(spellId);
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

void NeuralBotMgr::InitBcRecorder()
{
    std::string path = sConfigMgr->GetOption<std::string>("NeuralBot.BcRecordPath", "");
    _bcRecordEvery = std::max<uint32>(1, sConfigMgr->GetOption<uint32>("NeuralBot.BcRecordEvery", 1));
    if (path.empty())
        return;

    _bcFile = fopen(path.c_str(), "ab");
    if (!_bcFile)
    {
        LOG_ERROR("module.neuralbot", "BC recorder: cannot open {} — recording disabled", path);
        return;
    }
    // Append to an existing stream without a footer — records are self-delimiting by
    // fixed NB_BC_RECORD_SIZE, so a partial tail (from a crash) is skipped by Python.
    LOG_INFO("module.neuralbot", "BC recorder: writing demonstrations to {} (every {}th action)", path, _bcRecordEvery);
}

void NeuralBotMgr::CloseBcRecorder()
{
    if (_bcFile)
    {
        fflush(_bcFile);
        fclose(_bcFile);
        _bcFile = nullptr;
        LOG_INFO("module.neuralbot", "BC recorder: closed after {} records", _bcSeq);
    }
}

void NeuralBotMgr::OnPlayerbotActionExecuted(Player* player, std::string const& actionName, ObjectGuid target)
{
    if (!_bcFile || !player)
        return;

    // Downsample for long measurement runs (1 = record every action).
    if (++_bcCounter % _bcRecordEvery != 0)
        return;

    NeuralBotBcRecord rec{};
    rec.botGuid = player->GetGUID().GetRawValue();
    rec.targetGuid = target.GetRawValue();
    rec.seq = _bcSeq++;
    std::strncpy(rec.name, actionName.c_str(), NB_BC_NAME_LEN - 1);
    rec.name[NB_BC_NAME_LEN - 1] = '\0';

    // Frame for the playerbot; reward tail stays zeroed (progress reconstructed
    // from self xp/money/level deltas in Python).
    BuildFrameFor(player, rec.frame, nullptr, nullptr);

    fwrite(&rec, sizeof(rec), 1, _bcFile);

    // Periodic flush so analysis can tail the file while recording is live.
    if ((_bcSeq & 0x3FF) == 0)
        fflush(_bcFile);
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
