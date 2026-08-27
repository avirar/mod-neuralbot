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

// Baseline class abilities a real level-1 character is born with (rank-1 only).
// Higher ranks and new abilities remain trainer-bought — the RL policy must learn to
// visit trainers (see NeuralBotInstance's trainer interaction). Mirrors the level-1
// subset of mod-playerbots' PlayerbotFactory::InitClassSpells().
static void LearnBaselineSpells(Player* player)
{
    // Player::Create learns class/race spells via skill-line ability auto-learn with
    // temporary=true (the player is not yet in world), so _SaveSpells skips them. At
    // login, _LoadSkills already has the skills saved from creation, so LearnDefaultSkills
    // skips re-learning the abilities — leaving the bot with zero spells. Convert the
    // temporary spells to NEW so the creation SaveToDB persists them.
    uint32 persisted = 0;
    for (auto& [spellId, spellState] : player->GetSpellMap())
    {
        if (spellState->State == PLAYERSPELL_TEMPORARY)
        {
            spellState->State = PLAYERSPELL_NEW;
            ++persisted;
        }
    }

    // A few rank-1 abilities are not in the skill-line auto-learn set (e.g. warlock's
    // Summon Imp is normally quest-granted); learn them explicitly as persistent.
    switch (player->getClass())
    {
        case CLASS_WARRIOR:
            player->learnSpell(78);    // Heroic Strike
            player->learnSpell(2457);  // Battle Stance
            break;
        case CLASS_PALADIN:
            player->learnSpell(21084); // Seal of Righteousness
            player->learnSpell(635);   // Holy Light
            break;
        case CLASS_HUNTER:
            player->learnSpell(2973);  // Raptor Strike
            player->learnSpell(75);    // Auto Shot
            break;
        case CLASS_ROGUE:
            player->learnSpell(1752);  // Sinister Strike
            player->learnSpell(2098);  // Eviscerate
            break;
        case CLASS_PRIEST:
            player->learnSpell(585);   // Smite
            player->learnSpell(2050);  // Lesser Heal
            break;
        case CLASS_SHAMAN:
            player->learnSpell(403);   // Lightning Bolt
            player->learnSpell(331);   // Healing Wave
            break;
        case CLASS_MAGE:
            player->learnSpell(133);   // Fireball
            player->learnSpell(168);   // Frost Armor
            break;
        case CLASS_WARLOCK:
            player->learnSpell(687);   // Demon Skin
            player->learnSpell(686);   // Shadow Bolt
            player->learnSpell(688);   // Summon Imp
            break;
        case CLASS_DRUID:
            player->learnSpell(5176);  // Wrath
            player->learnSpell(5185);  // Healing Touch
            break;
        default:
            break;
    }

    LOG_INFO("module.neuralbot", "Baseline spells for '{}' (class {}): {} temporary -> new",
        player->GetName(), uint32(player->getClass()), persisted);
}

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
    // "Neuralbot" + 1..N base-26 letters, skipping names AzerothCore rejects:
    //  - names ending in "GM"/"gm" (reserved; ObjectMgr::IsReservedName hardcodes it)
    //  - names with three consecutive identical letters (case-insensitive;
    //    ObjectMgr::CheckPlayerName -> CHAR_NAME_THREE_CONSECUTIVE). Note the prefix
    //    "Neuralbot" ends in 't', so a suffix starting "TT" would make "ttt".
    auto encode = [](uint32 v, uint8 len) {
        std::string s(len, 'A');
        for (int8 k = static_cast<int8>(len) - 1; k >= 0; --k) { s[k] = static_cast<char>('A' + (v % 26)); v /= 26; }
        return s;
    };
    auto isValid = [](std::string const& n) {
        if (n.size() >= 2) {
            char a = static_cast<char>(std::tolower(static_cast<unsigned char>(n[n.size() - 2])));
            char b = static_cast<char>(std::tolower(static_cast<unsigned char>(n[n.size() - 1])));
            if (a == 'g' && b == 'm')
                return false;
        }
        for (size_t i = 2; i < n.size(); ++i) {
            char a = static_cast<char>(std::tolower(static_cast<unsigned char>(n[i])));
            char b = static_cast<char>(std::tolower(static_cast<unsigned char>(n[i - 1])));
            char c = static_cast<char>(std::tolower(static_cast<unsigned char>(n[i - 2])));
            if (a == b && b == c)
                return false;
        }
        return true;
    };

    // Map `index` to the index-th VALID name by counting valid candidates across
    // increasing name lengths. Deterministic; capacity ~180k names for 6 letters.
    uint32 seen = 0;
    for (uint8 len = 1; len <= 6; ++len)
    {
        uint32 cap = 1;
        for (uint8 k = 0; k < len; ++k)
            cap *= 26;
        for (uint32 v = 0; v < cap; ++v)
        {
            std::string name = "Neuralbot" + encode(v, len);
            if (!isValid(name))
                continue;
            if (seen == index)
                return name;
            ++seen;
        }
    }
    return "Neuralbot" + std::to_string(index); // unreachable for sane indices
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
    std::string prefix = sConfigMgr->GetOption<std::string>("NeuralBot.BotAccountPrefix", "nbot");
    for (uint32 i = 0; i < numAccounts; ++i)
    {
        std::string name = prefix + std::to_string(i);
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
        std::string accountName = sConfigMgr->GetOption<std::string>("NeuralBot.BotAccountPrefix", "nbot") + std::to_string(accountIdx);
        uint32 accountId = sAccountMgr->GetId(accountName);
        if (!accountId)
        {
            LOG_ERROR("module.neuralbot", "Account '{}' does not exist", accountName);
            return false;
        }

        // Check if character already exists — preserve across restarts (natural
        // progression: levels/quests/gear persist). NeuralBot.CleanupOnStartup=1 calls
        // CleanupBots() first for a fresh slate.
        CharacterCacheEntry const* entry = sCharacterCache->GetCharacterCacheByName(tpl.name);
        if (entry)
        {
            LOG_INFO("module.neuralbot", "Character '{}' already exists (GUID: {}, level {}), reusing",
                tpl.name, entry->Guid.GetCounter(), entry->Level);
            s_createdCharacters.push_back({tpl.name, entry->Guid, accountId});
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

        // Real level-1 characters are born with rank-1 abilities; learn them here
        // (once, at creation) so bots aren't unrealistically gimped. Everything else
        // is trainer-bought and must be learned by the policy.
        LearnBaselineSpells(player);

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
