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

class NeuralBotFactory
{
public:
    static std::vector<BotCharacterTemplate> GetBotTemplates();
    static bool CreateAccounts();
    static bool CreateCharacters();
};

#endif
