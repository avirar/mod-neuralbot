#include "NeuralBotMgr.h"
#include "NeuralBotFactory.h"
#include "NeuralBotCommon.h"
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

    LOG_INFO("module.neuralbot", "NeuralBot manager initialized. Target: {} bots", NeuralBotFactory::GetBotTemplates().size());
}

void NeuralBotMgr::Shutdown()
{
    _enabled = false;
    for (auto& [name, inst] : _instances)
        delete inst;
    _instances.clear();
    _instancesByGuid.clear();
}

void NeuralBotMgr::SpawnAndLoginBots()
{
    // Create characters first (requires fully initialized world)
    if (!NeuralBotFactory::CreateCharacters())
    {
        LOG_ERROR("module.neuralbot", "Failed to create characters");
        return;
    }

    // Wait for all DB writes to commit before any login attempt
    // (mod-playerbots pattern: without this, queries hit stale connections)
    while (CharacterDatabase.QueueSize())
        std::this_thread::sleep_for(1s);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Test with old Neuralbot first
    {
        QueryResult dbResult = CharacterDatabase.Query("SELECT guid, account FROM characters WHERE name = 'Neuralbot'");
        if (dbResult)
        {
            Field* f = dbResult->Fetch();
            ObjectGuid guid = ObjectGuid::Create<HighGuid::Player>(f[0].Get<uint32>());
            uint32 accountId = f[1].Get<uint32>();
            WorldSession* ts = new WorldSession(accountId, "", 0x0, nullptr, SEC_PLAYER,
                EXPANSION_WRATH_OF_THE_LICH_KING, time_t(0), sWorld->GetDefaultDbcLocale(),
                0, false, false, 0, true);
            _pendingLogins.push_back({accountId, guid, "Neuralbot_test", ts});
            LOG_INFO("module.neuralbot", "Added test Neuralbot to pending logins");
        }
    }

    auto created = NeuralBotFactory::GetCreatedCharacters();

    for (auto const& info : created)
    {
        // Create bot session (never registered with WorldSessionMgr)
        WorldSession* session = new WorldSession(info.accountId, "", 0x0, nullptr,
            SEC_PLAYER, EXPANSION_WRATH_OF_THE_LICH_KING, time_t(0),
            sWorld->GetDefaultDbcLocale(), 0, false, false, 0, true);

        auto holder = std::make_shared<LoginQueryHolder>(info.accountId, info.guid);
        if (!holder->Initialize())
        {
            LOG_ERROR("module.neuralbot", "Failed to init LoginQueryHolder for '{}'", info.name);
            delete session;
            continue;
        }

        _pendingLogins.push_back({info.accountId, info.guid, info.name, session});
    }

    LOG_INFO("module.neuralbot", "Queued {} bot(s) for async login", _pendingLogins.size());

    // Start staggered login process (first login after 5s delay)
    _loginScheduled = true;
    _loginTimer = 5000;
}

void NeuralBotMgr::OnWorldUpdate(uint32 diff)
{
    if (!_enabled)
        return;

    // Handle staggered login
    if (_loginScheduled && _pendingLoginIndex < _pendingLogins.size())
    {
        if (diff >= _loginTimer)
            _loginTimer = 0;
        else
            _loginTimer -= diff;

        if (_loginTimer == 0)
        {
            LOG_INFO("module.neuralbot", "Login timer fired, calling DoPendingLogin (idx={}/{})",
                _pendingLoginIndex, _pendingLogins.size());
            DoPendingLogin();
        }
    }

    // Process queued step/reset requests on the world thread
    // Loop to drain requests that arrive between world ticks
    while (ProcessPendingRequests())
    {
    }

    // Heartbeat for already-logged-in bots
    for (auto& [name, inst] : _instances)
        inst->ProcessBotPackets();
}

void NeuralBotMgr::DoPendingLogin()
{
    if (_pendingLoginIndex >= _pendingLogins.size())
    {
        _loginScheduled = false;
        LOG_INFO("module.neuralbot", "All {} bot(s) logged in successfully", _instances.size());
        return;
    }

    auto& pending = _pendingLogins[_pendingLoginIndex];
    _pendingLoginIndex++;

    LOG_INFO("module.neuralbot", "Logging in '{}' (GUID:{} Account:{})...", pending.name, pending.guid.GetCounter(), pending.accountId);

    auto holder = std::make_shared<LoginQueryHolder>(pending.accountId, pending.guid);
    if (!holder->Initialize())
    {
        LOG_ERROR("module.neuralbot", "Failed to init LoginQueryHolder for '{}'", pending.name);
        _loginTimer = 500;
        return;
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
                _loginTimer = 1500;
                return;
            }
            catch (...)
            {
                LOG_ERROR("module.neuralbot", "Unknown login exception for '{}'", name);
                _loginTimer = 1500;
                return;
            }

            Player* player = session->GetPlayer();
            if (!player)
            {
                LOG_ERROR("module.neuralbot", "Login failed for '{}' (no Player)", name);
                _loginTimer = 1500;
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

            _loginTimer = 2000;
        });
}

void NeuralBotMgr::OnPlayerLogin(Player* player)
{
    if (!_enabled || !player || !player->GetSession()->IsBot())
        return;

    // Check if this is one of our bots (not registered yet)
    std::string name = player->GetName();
    if (_instances.find(name) != _instances.end())
        return; // already registered

    // Create a new instance for this pre-logged-in bot
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

void NeuralBotMgr::OnPlayerCreatureKill(Player* killer, Creature* /*killed*/)
{
    if (!_enabled || !killer) return;
    auto it = _instancesByGuid.find(killer->GetGUID());
    if (it != _instancesByGuid.end())
        it->second->OnPlayerCreatureKill();
}

void NeuralBotMgr::OnPlayerAfterUpdate(Player* player, uint32 /*diff*/)
{
    if (!_enabled || !player) return;

    // Process bot packets for this player (called per-player from world thread)
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
    std::vector<std::string> names;
    for (auto const& [name, inst] : _instances)
        names.push_back(name);
    return names;
}
