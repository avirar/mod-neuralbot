#ifndef NEURALBOTMGR_H
#define NEURALBOTMGR_H

#include "NeuralBotCommon.h"
#include "NeuralBotInstance.h"
#include "Player.h"
#include "WorldSession.h"
#include "WorldPacket.h"

#include <mutex>
#include <deque>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <set>
#include <map>
#include <array>
#include <string>

class NeuralBotMgr
{
public:
    static NeuralBotMgr& instance();

    void Initialize();
    void Shutdown();

    bool IsEnabled() const { return _enabled; }

    void OnWorldUpdate(uint32 diff);
    void OnPlayerbotPacketSent(Player* player, WorldPacket const* packet);
    void OnPlayerLogin(Player* player);
    void OnPlayerJustDied(Player* player);
    void OnPlayerCreatureKill(Player* killer, Creature* killed);
    void OnPlayerAfterUpdate(Player* player, uint32 diff);

    NeuralBotStepResult Step(std::string const& botName, uint32 action);
    NeuralBotObservation Reset(std::string const& botName);
    void RecordOpcodeFor(Player* player, uint16 opcode);

    NeuralBotInstance* GetInstance(std::string const& botName);
    std::vector<std::string> GetBotNames() const;
    size_t GetBotCount() const { return _instances.size(); }
    void SpawnAndLoginBots();

private:
    NeuralBotMgr() = default;
    NeuralBotMgr(NeuralBotMgr const&) = delete;
    NeuralBotMgr& operator=(NeuralBotMgr const&) = delete;

    void DoPendingLogin();

    bool _enabled = false;

    std::map<std::string, NeuralBotInstance*> _instances;
    std::map<ObjectGuid, NeuralBotInstance*> _instancesByGuid;

    struct PendingLogin
    {
        uint32 accountId;
        ObjectGuid guid;
        std::string name;
        WorldSession* session;
    };
    std::vector<PendingLogin> _pendingLogins;
    size_t _pendingLoginIndex = 0;
    uint32 _loginTimer = 0;
    bool _loginScheduled = false;
};

#define sNeuralBotMgr NeuralBotMgr::instance()

#endif
