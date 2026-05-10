#include "NeuralBotFactory.h"
#include "CharacterCache.h"
#include "Player.h"
#include "WorldSession.h"
#include "ObjectMgr.h"
#include "AccountMgr.h"
#include "DatabaseEnv.h"
#include "Config.h"
#include "Log.h"

#include <vector>
#include <string>
#include <thread>
#include <tuple>

static std::vector<CreatedCharacterInfo> s_createdCharacters;

// All valid WoW race/class starter combos (letters-only names)
static const std::pair<uint8, uint8> STARTER_COMBOS[] = {
    // Alliance (5 races)
    {RACE_HUMAN,      CLASS_WARRIOR},
    {RACE_HUMAN,      CLASS_PRIEST},
    {RACE_HUMAN,      CLASS_MAGE},
    {RACE_HUMAN,      CLASS_PALADIN},
    {RACE_HUMAN,      CLASS_WARLOCK},
    {RACE_DWARF,      CLASS_HUNTER},
    {RACE_DWARF,      CLASS_PALADIN},
    {RACE_DWARF,      CLASS_ROGUE},
    {RACE_DWARF,      CLASS_PRIEST},
    {RACE_GNOME,      CLASS_ROGUE},
    {RACE_GNOME,      CLASS_MAGE},
    {RACE_GNOME,      CLASS_WARRIOR},
    {RACE_NIGHTELF,   CLASS_DRUID},
    {RACE_NIGHTELF,   CLASS_WARRIOR},
    {RACE_NIGHTELF,   CLASS_HUNTER},
    {RACE_NIGHTELF,   CLASS_ROGUE},
    {RACE_DRAENEI,    CLASS_SHAMAN},
    {RACE_DRAENEI,    CLASS_PALADIN},
    {RACE_DRAENEI,    CLASS_WARRIOR},
    {RACE_DRAENEI,    CLASS_MAGE},
    // Horde (5 races)
    {RACE_ORC,        CLASS_WARRIOR},
    {RACE_ORC,        CLASS_SHAMAN},
    {RACE_ORC,        CLASS_HUNTER},
    {RACE_ORC,        CLASS_ROGUE},
    {RACE_TROLL,      CLASS_HUNTER},
    {RACE_TROLL,      CLASS_MAGE},
    {RACE_TROLL,      CLASS_PRIEST},
    {RACE_TAUREN,     CLASS_WARRIOR},
    {RACE_TAUREN,     CLASS_DRUID},
    {RACE_TAUREN,     CLASS_SHAMAN},
    {RACE_TAUREN,     CLASS_HUNTER},
    {RACE_UNDEAD_PLAYER, CLASS_ROGUE},
    {RACE_UNDEAD_PLAYER, CLASS_PRIEST},
    {RACE_UNDEAD_PLAYER, CLASS_WARRIOR},
    {RACE_UNDEAD_PLAYER, CLASS_MAGE},
    {RACE_BLOODELF,   CLASS_WARLOCK},
    {RACE_BLOODELF,   CLASS_PALADIN},
    {RACE_BLOODELF,   CLASS_HUNTER},
    {RACE_BLOODELF,   CLASS_MAGE},
};
constexpr size_t NUM_COMBOS = sizeof(STARTER_COMBOS) / sizeof(STARTER_COMBOS[0]);

uint32 NeuralBotFactory::GetBotCount()
{
    return sConfigMgr->GetOption<uint32>("NeuralBot.BotCount", 20);
}

std::string NeuralBotFactory::GenerateBotName(uint32 index)
{
    if (index < 26)
        return std::string("Neuralbot") + static_cast<char>('A' + index);
    uint32 i = index - 26;
    char first  = static_cast<char>('A' + i / 26);
    char second = static_cast<char>('A' + i % 26);
    return std::string("Neuralbot") + first + second;
}

std::vector<BotCharacterTemplate> NeuralBotFactory::GetBotTemplates()
{
    uint32 botCount = GetBotCount();
    std::vector<BotCharacterTemplate> templates;
    templates.reserve(botCount);
    for (uint32 i = 0; i < botCount; ++i)
    {
        auto const& combo = STARTER_COMBOS[i % NUM_COMBOS];
        templates.push_back({GenerateBotName(i), combo.first, combo.second,
                             static_cast<uint8>(i % 2 ? GENDER_FEMALE : GENDER_MALE)});
    }
    return templates;
}

bool NeuralBotFactory::CreateAccounts()
{
    uint32 botCount = GetBotCount();
    uint32 numAccounts = (botCount + 9) / 10; // 10 chars per account (WoW limit)
    for (uint32 i = 0; i < numAccounts; ++i)
    {
        std::string name = "nbot" + std::to_string(i);
        uint32 existingId = sAccountMgr->GetId(name);
        if (existingId)
        {
            LOG_INFO("module.neuralbot", "Account '{}' already exists (ID: {})", name, existingId);
            continue;
        }
        if (sAccountMgr->CreateAccount(name, name) != AOR_OK)
        {
            LOG_ERROR("module.neuralbot", "Failed to create account '{}'", name);
            return false;
        }
        LOG_INFO("module.neuralbot", "Created account '{}'", name);
    }

    while (LoginDatabase.QueueSize())
        std::this_thread::sleep_for(1s);

    return true;
}

bool NeuralBotFactory::CreateCharacters()
{
    s_createdCharacters.clear();
    auto templates = GetBotTemplates();

    for (size_t i = 0; i < templates.size(); ++i)
    {
        auto const& tpl = templates[i];

        // Determine which account this bot belongs to
        uint32 accountIdx = static_cast<uint32>(i) / 10;
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
            LOG_INFO("module.neuralbot", "Character '{}' already exists (GUID: {}), force-recreating", tpl.name, entry->Guid.GetCounter());
            CharacterDatabase.Execute("DELETE FROM characters WHERE guid = " + std::to_string(entry->Guid.GetCounter()));
            sCharacterCache->DeleteCharacterCacheEntry(entry->Guid, entry->Name);
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

        s_createdCharacters.push_back({tpl.name, player->GetGUID(), accountId});

        LOG_INFO("module.neuralbot", "Created character '{}' (race={}, class={}, GUID={})",
            tpl.name, uint32(tpl.race), uint32(tpl.cls), guid);

        player->CleanupsBeforeDelete();
        delete player;
        delete session;
    }

    return true;
}

std::vector<CreatedCharacterInfo> const& NeuralBotFactory::GetCreatedCharacters()
{
    return s_createdCharacters;
}

void NeuralBotFactory::CleanupBots()
{
    LOG_INFO("module.neuralbot", "Cleaning up all NeuralBot characters and accounts...");

    // Find bot account IDs
    QueryResult accounts = LoginDatabase.Query(
        "SELECT id, username FROM account WHERE username REGEXP '^nbot[0-9]+$'");
    if (!accounts)
    {
        LOG_INFO("module.neuralbot", "No bot accounts found to clean.");
        return;
    }

    std::vector<uint32> accountIds;
    do
    {
        Field* f = accounts->Fetch();
        uint32 id = f[0].Get<uint32>();
        accountIds.push_back(id);
        LOG_INFO("module.neuralbot", "Found bot account '{}' (ID: {})", f[1].Get<std::string>(), id);
    } while (accounts->NextRow());

    // Build account ID list for SQL IN clause
    std::string accountList;
    for (size_t i = 0; i < accountIds.size(); ++i)
    {
        if (i > 0) accountList += ",";
        accountList += std::to_string(accountIds[i]);
    }

    // Delete characters belonging to bot accounts
    CharacterDatabase.Execute("DELETE FROM characters WHERE account IN (" + accountList + ")");
    LOG_INFO("module.neuralbot", "Deleted bot characters, waiting for DB queue to drain...");

    while (CharacterDatabase.QueueSize())
        std::this_thread::sleep_for(1s);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Clean orphaned dependent rows (tables with guid column)
    LOG_INFO("module.neuralbot", "Cleaning orphaned dependent data...");
    CharacterDatabase.Execute("DELETE FROM character_account_data WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_achievement WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_achievement_progress WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_action WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_arena_stats WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_aura WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_banned WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_battleground_random WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_brew_of_the_month WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_declinedname WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_entry_point WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_equipmentsets WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_gifts WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_glyphs WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_homebind WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_instance WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_inventory WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_queststatus WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_queststatus_daily WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_queststatus_monthly WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_queststatus_rewarded WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_queststatus_seasonal WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_queststatus_weekly WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_reputation WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_settings WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_skills WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_social WHERE guid NOT IN (SELECT guid FROM characters) OR friend NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_spell WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_spell_cooldown WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_stats WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_talent WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_achievement_offline_updates WHERE guid NOT IN (SELECT guid FROM characters)");

    // Clean item_instance, corpse, mail, pet, etc.
    CharacterDatabase.Execute("DELETE FROM item_instance WHERE owner_guid NOT IN (SELECT guid FROM characters) AND owner_guid > 0");
    CharacterDatabase.Execute("DELETE FROM corpse WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM character_pet WHERE owner NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM mail WHERE receiver NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM mail_items WHERE receiver NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM petition WHERE ownerguid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM petition_sign WHERE ownerguid NOT IN (SELECT guid FROM characters) OR playerguid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM arena_team_member WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM gm_survey WHERE guid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM lag_reports WHERE guid NOT IN (SELECT guid FROM characters)");

    // Clean pet dependents
    CharacterDatabase.Execute("DELETE FROM pet_aura WHERE guid NOT IN (SELECT id FROM character_pet)");
    CharacterDatabase.Execute("DELETE FROM pet_spell WHERE guid NOT IN (SELECT id FROM character_pet)");
    CharacterDatabase.Execute("DELETE FROM pet_spell_cooldown WHERE guid NOT IN (SELECT id FROM character_pet)");

    // Clean group data
    CharacterDatabase.Execute("DELETE FROM `groups` WHERE leaderGuid NOT IN (SELECT guid FROM characters)");
    CharacterDatabase.Execute("DELETE FROM group_member WHERE memberGuid NOT IN (SELECT guid FROM characters)");

    // Wait for cleanup to drain
    while (CharacterDatabase.QueueSize())
        std::this_thread::sleep_for(1s);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Delete bot accounts
    LOG_INFO("module.neuralbot", "Deleting bot accounts...");
    for (uint32 accId : accountIds)
        AccountMgr::DeleteAccount(accId);

    // Wait and flush
    while (LoginDatabase.QueueSize() || CharacterDatabase.QueueSize())
        std::this_thread::sleep_for(1s);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    LOG_INFO("module.neuralbot", "Bot cleanup complete. Shutting down.");
}
