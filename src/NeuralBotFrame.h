#ifndef NEURALBOTFRAME_H
#define NEURALBOTFRAME_H

// Faithful structured observation frame (shm protocol v2).
//
// The per-bot observation is a single fixed-size, byte-packed struct exchanged over
// shared memory. All values are REAL game values (health, position, level, money, …) —
// no normalization, no hand-picked scalars. Sections use fixed caps so the mmap region
// stays precomputable and numpy can view it zero-copy:
//
//   self (fixed) | target (fixed) | counts (fixed)
//   spells[64]   | quests[16]     | entities[64]  | items[16]
//   reward tail (total + 14 diagnostic components)
//
// Wire structs are packed (1-byte alignment). Python mirrors them with numpy structured
// dtypes (align=False) — the byte layouts must match exactly. `SHM_VERSION` = 2 and the
// control block carries `frame_bytes` so the two sides negotiate.

#include <cstdint>
#include <string>

constexpr size_t NB_MAX_SPELLS   = 64;
constexpr size_t NB_MAX_QUESTS   = 16;
constexpr size_t NB_MAX_ENTITIES = 64;
constexpr size_t NB_MAX_ITEMS    = 16;
constexpr size_t NB_REWARD_COMPONENTS = 14;

// Entity kinds (mirrors python/neuralbot_client.py)
constexpr uint8_t NB_ENTITY_TYPE_NONE       = 0;
constexpr uint8_t NB_ENTITY_TYPE_CREATURE   = 1;
constexpr uint8_t NB_ENTITY_TYPE_PLAYER     = 2;
constexpr uint8_t NB_ENTITY_TYPE_GAMEOBJECT = 3;

// Reaction (real faction model: IsHostileTo / IsFriendlyTo, else neutral)
constexpr uint8_t NB_REACTION_NEUTRAL  = 0;
constexpr uint8_t NB_REACTION_HOSTILE  = 1;
constexpr uint8_t NB_REACTION_FRIENDLY = 2;

#pragma pack(push, 1)

// Fixed self record: real player state.
struct NBStateSelf
{
    uint64_t guid;
    uint32_t level;
    float health;
    float maxHealth;
    float mana;
    float maxMana;
    float resource;       // class secondary: rage/energy/runic/focus
    float maxResource;
    uint32_t xp;
    uint32_t nextLevelXp;
    uint32_t money;       // copper
    float posX;
    float posY;
    float posZ;
    float orientation;
    uint32_t mapId;
    uint32_t zoneId;
    uint32_t areaId;
    uint8_t alive;
    uint8_t inCombat;
    uint8_t moving;
    uint8_t casting;
    uint8_t inWater;
    uint8_t mounted;
    uint8_t classId;
    uint8_t race;
    uint32_t comboPoints;
    uint64_t targetGuid;
};

// Fixed current-target record (zeroed when no target).
struct NBStateTarget
{
    uint64_t guid;
    uint32_t entry;
    uint8_t type;         // NB_ENTITY_TYPE_*
    float health;
    float maxHealth;
    uint32_t level;
    float dx;             // relative to self
    float dy;
    float dz;
    float distance;
    uint8_t reaction;     // NB_REACTION_*
    uint8_t alive;
    uint8_t inCombat;
    uint8_t casting;
    uint32_t npcFlags;
};

// Counts of populated records in each variable section.
struct NBStateCounts
{
    uint16_t nSpells;
    uint16_t nQuests;
    uint16_t nEntities;
    uint16_t nItems;
};

// One spellbook entry.
struct NBSpellRec
{
    uint32_t spellId;
    uint32_t cooldownMs;   // remaining cooldown, 0 if ready
    uint32_t cost;         // primary power cost
    float range;           // max range
    float minRange;
    float castTimeMs;
    uint8_t ready;
    uint8_t _pad[3];
};

// One quest slot.
struct NBQuestRec
{
    uint32_t questId;
    uint8_t status;        // QuestStatus
    uint8_t _pad[3];
    uint16_t obj[4];       // objective progress counters
};

// One nearby entity (creature / player / gameobject).
struct NBEntityRec
{
    uint64_t guid;
    uint32_t entry;
    uint8_t type;          // NB_ENTITY_TYPE_*
    uint8_t _pad[3];
    uint32_t level;
    float health;
    float maxHealth;
    float dx;              // relative to self
    float dy;
    float dz;
    float distance;
    uint8_t reaction;      // NB_REACTION_*
    uint8_t alive;
    uint8_t inCombat;
    uint8_t casting;
    uint32_t npcFlags;
};

// One nearby ground item / lootable object.
struct NBItemRec
{
    uint64_t guid;
    uint32_t entry;
    uint8_t quality;
    uint8_t _pad[3];
    float distance;
};

// Reward tail: native total + diagnostic components (not summed; analysis only).
struct NBStateReward
{
    float total;
    float components[NB_REWARD_COMPONENTS];
};

struct NeuralBotFrame
{
    NBStateSelf self;
    NBStateTarget target;
    NBStateCounts counts;
    NBSpellRec spells[NB_MAX_SPELLS];
    NBQuestRec quests[NB_MAX_QUESTS];
    NBEntityRec entities[NB_MAX_ENTITIES];
    NBItemRec items[NB_MAX_ITEMS];
    NBStateReward reward;
};

#pragma pack(pop)

struct NeuralBotFrameResult
{
    NeuralBotFrame frame;
    bool done = false;
    std::string info;
};

static_assert(sizeof(NBStateSelf) == 96, "NBStateSelf layout changed — mirror in Python");
static_assert(sizeof(NBStateTarget) == 49, "NBStateTarget layout changed — mirror in Python");
static_assert(sizeof(NBStateCounts) == 8, "NBStateCounts layout changed — mirror in Python");
static_assert(sizeof(NBSpellRec) == 28, "NBSpellRec layout changed — mirror in Python");
static_assert(sizeof(NBQuestRec) == 16, "NBQuestRec layout changed — mirror in Python");
static_assert(sizeof(NBEntityRec) == 52, "NBEntityRec layout changed — mirror in Python");
static_assert(sizeof(NBItemRec) == 20, "NBItemRec layout changed — mirror in Python");
static_assert(sizeof(NBStateReward) == 60, "NBStateReward layout changed — mirror in Python");

#endif // NEURALBOTFRAME_H
