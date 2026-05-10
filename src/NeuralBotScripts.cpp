#include "NeuralBotMgr.h"
#include "NeuralBotWSHandler.h"
#include "NeuralBotCommon.h"
#include "ScriptMgr.h"
#include "Config.h"
#include "Player.h"
#include "Creature.h"
#include "World.h"
#include "Log.h"

class NeuralBotWorldScript : public WorldScript
{
public:
    NeuralBotWorldScript() : WorldScript("NeuralBotWorldScript") {}

    void OnAfterConfigLoad(bool /*reload*/) override
    {
        sNeuralBotMgr.Initialize();
    }

    void OnStartup() override
    {
        if (sNeuralBotMgr.IsEnabled())
        {
            uint16 port = sConfigMgr->GetOption<uint16>("NeuralBot.WebSocketPort", 9000);
            sNeuralBotWS.Start(port);
            sNeuralBotMgr.ScheduleLogin();
            LOG_INFO("module.neuralbot", "NeuralBot world script started");
        }
    }

    void OnUpdate(uint32 diff) override
    {
        sNeuralBotMgr.OnWorldUpdate(diff);
    }
};

class NeuralBotPlayerScript : public PlayerScript
{
public:
    NeuralBotPlayerScript() : PlayerScript("NeuralBotPlayerScript", {
        PLAYERHOOK_ON_LOGIN,
        PLAYERHOOK_ON_AFTER_UPDATE,
        PLAYERHOOK_ON_PLAYER_JUST_DIED,
        PLAYERHOOK_ON_CREATURE_KILL
    }) {}

    void OnPlayerLogin(Player* player) override
    {
        sNeuralBotMgr.OnPlayerLogin(player);
    }

    void OnPlayerAfterUpdate(Player* player, uint32 diff) override
    {
        sNeuralBotMgr.OnPlayerAfterUpdate(player, diff);
    }

    void OnPlayerJustDied(Player* player) override
    {
        sNeuralBotMgr.OnPlayerJustDied(player);
    }

    void OnPlayerCreatureKill(Player* killer, Creature* killed) override
    {
        sNeuralBotMgr.OnPlayerCreatureKill(killer, killed);
    }
};

class NeuralBotPlayerbotScript : public PlayerbotScript
{
public:
    NeuralBotPlayerbotScript() : PlayerbotScript("NeuralBotPlayerbotScript") {}

    void OnPlayerbotPacketSent(Player* player, WorldPacket const* packet) override
    {
        sNeuralBotMgr.OnPlayerbotPacketSent(player, packet);
    }
};

class NeuralBotServerScript : public ServerScript
{
public:
    NeuralBotServerScript() : ServerScript("NeuralBotServerScript") {}
};

void AddNeuralBotScripts()
{
    new NeuralBotWorldScript();
    new NeuralBotPlayerScript();
    new NeuralBotPlayerbotScript();
    new NeuralBotServerScript();
}
