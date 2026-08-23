"""
Map recorded playerbot actions (frame, actionName, targetGuid) to our 41-action space.

The recorded action names are the playerbot engine's string-named actions (see
mod-playerbots `src/Ai/Base/Actions/`). Only the *valuable, world-level* subset maps
cleanly onto our discrete actions; meta/status actions ("drink", "xp gain",
"suggest what to do", "set facing", ...) produce no progress and are skipped — this is
the return-filtering from the review's recommendation (clone the competent combat/loot/
quest behaviors, not the engine's bookkeeping noise).
"""
import struct

# action ids (mirror neuralbot_client.ACTION_*)
A_MOVE_TO_TARGET = 1
A_ATTACK_START = 28
A_CAST_0 = 30
A_INTERACT = 38
A_LOOT = 40

MAX_CAST_SLOTS = 8

ATTACK_NAMES = {
    "attack anything", "attack", "attack target", "dps assist", "melee",
    "reach melee", "pull", "auto attack",
}
LOOT_NAMES = {
    "add all loot", "open loot", "loot", "store loot", "add gathering loot",
    "find corpse", "gather",
}
MOVE_NAMES = {
    "new rpg go grind", "new rpg go camp", "new rpg move npcs", "reach spell",
    "reach target", "follow", "move to loot", "move random",
}
QUEST_NAMES = {
    "new rpg do quest", "accept quest", "turn in quest", "talk to quest giver",
    "gossip", "complete quest", "interact with npc",
}
# Engine bookkeeping / non-world actions: skip (they carry ~0 progress).
SKIP_NAMES = {
    "drink", "drop target", "xp gain", "suggest what to do", "set facing",
    "check mount state", "new rpg status update", "new rpg update",
    "use trinket", "apply oil", "apply stone", "food", "unstealth", "lfg leave",
    "new rpg travel flight", "flee", "toggle pet spell", "set pet stance",
    "new rpg update objective", "move from corpse", "new rpg go hide",
}

# Spell.dbc: name field is field 136 (the first 's' in SpellEntryfmt); each field is 4 B.
SPELL_NAME_FIELD_OFFSET = 136 * 4


def load_spell_name_map(dbc_path: str) -> dict:
    """Parse Spell.dbc once into {lowercase_name: spellId}."""
    with open(dbc_path, "rb") as f:
        assert f.read(4) == b"WDBC", "not a WDBC file"
        rec_count, _field_count, rec_size, str_size = struct.unpack("<4I", f.read(16))
        data = f.read(rec_count * rec_size)
        strblock = f.read(str_size)

    name_map = {}
    for i in range(rec_count):
        rec = data[i * rec_size:(i + 1) * rec_size]
        sid = struct.unpack_from("<I", rec, 0)[0]
        noff = struct.unpack_from("<I", rec, SPELL_NAME_FIELD_OFFSET)[0]
        if noff == 0:
            continue
        end = strblock.find(b"\x00", noff)
        if end < 0:
            continue
        name = strblock[noff:end].decode("utf-8", "replace").strip().lower()
        if name and name not in name_map:
            name_map[name] = sid
    return name_map


def build_mapper(dbc_path: str):
    """Return a function (frame, action_name, target_guid) -> action_id | None."""
    spell_ids = load_spell_name_map(dbc_path)

    def map_action(frame, name, target_guid) -> int | None:
        n = name.strip().lower()
        if n in SKIP_NAMES:
            return None
        if n in ATTACK_NAMES:
            return A_ATTACK_START
        if n in LOOT_NAMES:
            return A_LOOT
        if n in MOVE_NAMES:
            return A_MOVE_TO_TARGET
        if n in QUEST_NAMES:
            return A_INTERACT
        # Spell cast: "corruption on attacker" -> base name "corruption".
        base = n.split(" on ", 1)[0]
        sid = spell_ids.get(base)
        if sid is None:
            return None
        # CAST_SPELL_i only reaches the first 8 spellbook entries. The C++ frame now
        # sorts spells by spellId (NB: historical recordings were unordered), so compute
        # the spell's index in spellId-sorted order to match the C++ semantics.
        full_ids = sorted(int(x) for x in frame["spells"]["spellId"] if int(x) != 0)
        if sid in full_ids:
            pos = full_ids.index(sid)
            if pos < MAX_CAST_SLOTS:
                return A_CAST_0 + pos
        return None  # spell known but not in the first 8 sorted slots — uncastable

    return map_action
