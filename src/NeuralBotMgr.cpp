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

    // Create accounts and characters via factory
    NeuralBotFactory::CreateAccounts();
    NeuralBotFactory::CreateCharacters();

    // Schedule bot login (async, staggered)
    _loginScheduled = true;
    _pendingLoginIndex = 0;
    _loginTimer = 2000; // start first login after 2s delay

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
    auto templates = NeuralBotFactory::GetBotTemplates();

    for (auto const& tpl : templates)
    {
        // Check if character exists in cache
        CharacterCacheEntry const* entry = sCharacterCache->GetCharacterCacheByName(tpl.name);
        if (!entry)
        {
            LOG_ERROR("module.neuralbot", "Character '{}' not found in cache", tpl.name);
            continue;
        }

        uint32 accountId = entry->AccountId;
        ObjectGuid guid = entry->Guid;

        // Create bot session (never registered with WorldSessionMgr)
        WorldSession* session = new WorldSession(accountId, "", 0x0, nullptr,
            SEC_PLAYER, EXPANSION_WRATH_OF_THE_LICH_KING, time_t(0),
            sWorld->GetDefaultDbcLocale(), 0, false, false, 0, true);

        auto holder = std::make_shared<LoginQueryHolder>(accountId, guid);
        if (!holder->Initialize())
        {
            LOG_ERROR("module.neuralbot", "Failed to init LoginQueryHolder for '{}'", tpl.name);
            delete session;
            continue;
        }

        _pendingLogins.push_back({accountId, guid, tpl.name, session});
    }

    LOG_INFO("module.neuralbot", "Queued {} bot(s) for async login", _pendingLogins.size());
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
            DoPendingLogin();
    }

    // Heartbeat for already-logged-in bots (process packets)
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

    auto holder = std::make_shared<LoginQueryHolder>(pending.accountId, pending.guid);
    if (!holder->Initialize())
    {
        LOG_ERROR("module.neuralbot", "Failed to init LoginQueryHolder for '{}'", pending.name);
        _loginTimer = 500;
        return;
    }

    WorldSession* session = pending.session;

    sWorld->AddQueryHolderCallback(CharacterDatabase.DelayQueryHolder(holder))
        .AfterComplete([this, session, name = pending.name](SQLQueryHolderBase const& queryHolder)
        {
            LoginQueryHolder const& lqh = static_cast<LoginQueryHolder const&>(queryHolder);

            try
            {
                session->HandlePlayerLoginFromDB(lqh);
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("module.neuralbot", "Login exception for '{}': {}", name, e.what());
                _loginTimer = 500;
                return;
            }

            Player* player = session->GetPlayer();
            if (!player)
            {
                LOG_ERROR("module.neuralbot", "Login failed for '{}'", name);
                _loginTimer = 500;
                return;
            }

            // Verify session is bot
            ASSERT(session->IsBot());

            // Create instance
            NeuralBotInstance* inst = new NeuralBotInstance(player, session);
            _instances[name] = inst;
            _instancesByGuid[player->GetGUID()] = inst;

            LOG_INFO("module.neuralbot", "Bot '{}' logged in (GUID:{} Level:{} Zone:{})",
                name, player->GetGUID().GetCounter(), uint32(player->GetLevel()), player->GetZoneId());

            // Schedule next bot login with 1.5s stagger to avoid DB contention
            _loginTimer = 1500;
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
    auto it = _instances.find(botName);
    if (it != _instances.end())
        return it->second->Step(action);
    NeuralBotStepResult result;
    result.done = true;
    result.info = "Bot not found: " + botName;
    return result;
}

NeuralBotObservation NeuralBotMgr::Reset(std::string const& botName)
{
    auto it = _instances.find(botName);
    if (it != _instances.end())
        return it->second->Reset();
    return NeuralBotObservation{};
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
