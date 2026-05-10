#include "NeuralBotFactory.h"
#include "CharacterCache.h"
#include "Player.h"
#include "WorldSession.h"
#include "ObjectMgr.h"
#include "AccountMgr.h"
#include "Log.h"

#include <vector>
#include <string>

std::vector<BotCharacterTemplate> NeuralBotFactory::GetBotTemplates()
{
    return {
        // Account nbot0: Alliance
        {"Neuralbot0",  RACE_HUMAN,      CLASS_WARRIOR,  GENDER_MALE},
        {"Neuralbot1",  RACE_HUMAN,      CLASS_PRIEST,   GENDER_FEMALE},
        {"Neuralbot2",  RACE_HUMAN,      CLASS_MAGE,     GENDER_MALE},
        {"Neuralbot3",  RACE_DWARF,      CLASS_HUNTER,   GENDER_MALE},
        {"Neuralbot4",  RACE_GNOME,      CLASS_ROGUE,    GENDER_FEMALE},
        {"Neuralbot5",  RACE_NIGHTELF,   CLASS_DRUID,    GENDER_FEMALE},
        {"Neuralbot6",  RACE_NIGHTELF,   CLASS_WARRIOR,  GENDER_MALE},
        {"Neuralbot7",  RACE_DRAENEI,    CLASS_SHAMAN,   GENDER_FEMALE},
        {"Neuralbot8",  RACE_DRAENEI,    CLASS_PALADIN,  GENDER_MALE},
        {"Neuralbot9",  RACE_DWARF,      CLASS_PALADIN,  GENDER_FEMALE},
        // Account nbot1: Horde
        {"Neuralbot10", RACE_ORC,        CLASS_WARRIOR,  GENDER_MALE},
        {"Neuralbot11", RACE_ORC,        CLASS_SHAMAN,   GENDER_MALE},
        {"Neuralbot12", RACE_TROLL,      CLASS_HUNTER,   GENDER_MALE},
        {"Neuralbot13", RACE_TAUREN,     CLASS_WARRIOR,  GENDER_MALE},
        {"Neuralbot14", RACE_TAUREN,     CLASS_DRUID,    GENDER_FEMALE},
        {"Neuralbot15", RACE_UNDEAD_PLAYER, CLASS_ROGUE, GENDER_MALE},
        {"Neuralbot16", RACE_UNDEAD_PLAYER, CLASS_PRIEST,GENDER_FEMALE},
        {"Neuralbot17", RACE_BLOODELF,   CLASS_WARLOCK,  GENDER_MALE},
        {"Neuralbot18", RACE_TAUREN,     CLASS_SHAMAN,   GENDER_FEMALE},
        {"Neuralbot19", RACE_ORC,        CLASS_HUNTER,   GENDER_FEMALE},
    };
}

bool NeuralBotFactory::CreateAccounts()
{
    for (uint32 i = 0; i < 2; ++i)
    {
        std::string name = "nbot" + std::to_string(i);
        uint32 existingId = sAccountMgr->GetId(name);
        if (existingId)
        {
            LOG_INFO("module.neuralbot", "Account '{}' already exists (ID: {})", name, existingId);
            continue;
        }
        if (!sAccountMgr->CreateAccount(name, name))
        {
            LOG_ERROR("module.neuralbot", "Failed to create account '{}'", name);
            return false;
        }
        LOG_INFO("module.neuralbot", "Created account '{}'", name);
    }
    return true;
}

bool NeuralBotFactory::CreateCharacters()
{
    auto templates = GetBotTemplates();
    std::vector<BotCharacterTemplate> remaining;

    for (auto const& tpl : templates)
    {
        // Determine which account this bot belongs to
        uint32 botNum = std::stoul(tpl.name.substr(9)); // "Neuralbot" -> num
        uint32 accountIdx = botNum < 10 ? 0 : 1;
        std::string accountName = "nbot" + std::to_string(accountIdx);
        uint32 accountId = sAccountMgr->GetId(accountName);
        if (!accountId)
        {
            LOG_ERROR("module.neuralbot", "Account '{}' does not exist", accountName);
            return false;
        }

        // Check if character already exists
        CharacterCacheEntry const* entry = sCharacterCache->GetCharacterCacheByName(tpl.name);
        if (entry)
        {
            LOG_INFO("module.neuralbot", "Character '{}' already exists (GUID: {})", tpl.name, entry->Guid.GetCounter());
            continue;
        }

        // Create temporary WorldSession for character creation
        WorldSession* session = new WorldSession(accountId, "", 0x0, nullptr, SEC_PLAYER,
            EXPANSION_WRATH_OF_THE_LICH_KING, time_t(0), sWorld->GetDefaultDbcLocale(),
            0, false, false, 0, true);

        auto charInfo = std::make_unique<CharacterCreateInfo>(
            tpl.name, tpl.race, tpl.cls, tpl.gender,
            0, 0, 0, 0, 0);

        Player* player = new Player(session);
        player->GetMotionMaster()->Initialize();

        ObjectGuid::LowType guid = sObjectMgr->GetGenerator<HighGuid::Player>().Generate();
        if (!player->Create(guid, charInfo.get()))
        {
            LOG_ERROR("module.neuralbot", "Failed to create character '{}'", tpl.name);
            delete player;
            delete session;
            return false;
        }

        player->setCinematic(2);
        player->SetAtLoginFlag(AT_LOGIN_NONE);
        player->SaveToDB(true, false);

        sCharacterCache->AddCharacterCacheEntry(player->GetGUID(), accountId,
            player->GetName(), player->getGender(), player->getRace(),
            player->getClass(), player->GetLevel());

        LOG_INFO("module.neuralbot", "Created character '{}' (race={}, class={}, GUID={})",
            tpl.name, uint32(tpl.race), uint32(tpl.cls), guid);

        player->CleanupsBeforeDelete();
        delete player;
        delete session;
    }

    return true;
}
