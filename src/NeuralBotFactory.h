#ifndef NEURALBOTFACTORY_H
#define NEURALBOTFACTORY_H

#include "NeuralBotCommon.h"
#include "ObjectGuid.h"
#include <vector>
#include <string>

struct BotCharacterTemplate
{
    std::string name;
    uint8 race;
    uint8 cls;
    uint8 gender;
};

struct CreatedCharacterInfo
{
    std::string name;
    ObjectGuid guid;
    uint32 accountId;
};

class NeuralBotFactory
{
public:
    static std::string GenerateBotName(uint32 index);
    static std::vector<BotCharacterTemplate> GetBotTemplates();
    static bool CreateAccounts();
    static bool CreateCharacters();
    static void CleanupBots();
    static std::vector<CreatedCharacterInfo> const& GetCreatedCharacters();
    static uint32 GetBotCount();
};

#endif
