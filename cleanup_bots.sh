#!/bin/bash
# Cleanup all NeuralBot characters and accounts.
# Run this with the worldserver OFFLINE.
# Usage: ./cleanup_bots.sh [mysql_user] [mysql_pass]
# Defaults: user=acore, pass=abc

set -euo pipefail

USER="${1:-acore}"
PASS="${2:-abc}"
AUTH_DB="acore_auth"
CHAR_DB="acore_characters"

echo "=== NeuralBot Cleanup ==="

# Find bot account IDs
echo "Looking up bot accounts..."
BOT_ACCOUNTS=$(mysql -u "$USER" -p"$PASS" "$AUTH_DB" -N -e "SELECT id FROM account WHERE username REGEXP '^nbot[0-9]+$'")
if [ -z "$BOT_ACCOUNTS" ]; then
    echo "No bot accounts found. Nothing to clean."
    exit 0
fi

echo "Found accounts: $(echo "$BOT_ACCOUNTS" | tr '\n' ' ')"
ACCOUNT_LIST=$(echo "$BOT_ACCOUNTS" | tr '\n' ',' | sed 's/,$//')

# Delete characters belonging to bot accounts
echo "Deleting bot characters..."
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM characters WHERE account IN ($ACCOUNT_LIST)"

echo "Cleaning orphaned character data..."
for table in \
    character_account_data character_achievement character_achievement_progress \
    character_action character_arena_stats character_aura character_banned \
    character_battleground_random character_brew_of_the_month character_declinedname \
    character_entry_point character_equipmentsets character_gifts character_glyphs \
    character_homebind character_instance character_inventory \
    character_queststatus character_queststatus_daily character_queststatus_monthly \
    character_queststatus_rewarded character_queststatus_seasonal character_queststatus_weekly \
    character_reputation character_settings character_skills character_social \
    character_spell character_spell_cooldown character_stats character_talent \
    character_achievement_offline_updates; do
    mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM $table WHERE guid NOT IN (SELECT guid FROM characters)" 2>/dev/null
done

echo "Cleaning items, corpses, pets, mail..."
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM item_instance WHERE owner_guid NOT IN (SELECT guid FROM characters) AND owner_guid > 0" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM corpse WHERE guid NOT IN (SELECT guid FROM characters)" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM character_pet WHERE owner NOT IN (SELECT guid FROM characters)" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM mail WHERE receiver NOT IN (SELECT guid FROM characters)" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM mail_items WHERE receiver NOT IN (SELECT guid FROM characters)" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM petition WHERE ownerguid NOT IN (SELECT guid FROM characters)" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM petition_sign WHERE ownerguid NOT IN (SELECT guid FROM characters) OR playerguid NOT IN (SELECT guid FROM characters)" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM arena_team_member WHERE guid NOT IN (SELECT guid FROM characters)" 2>/dev/null

echo "Cleaning pet dependents..."
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM pet_aura WHERE guid NOT IN (SELECT id FROM character_pet)"
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM pet_spell WHERE guid NOT IN (SELECT id FROM character_pet)"
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM pet_spell_cooldown WHERE guid NOT IN (SELECT id FROM character_pet)"

echo "Cleaning group data..."
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM \`groups\` WHERE leaderGuid NOT IN (SELECT guid FROM characters)" 2>/dev/null
mysql -u "$USER" -p"$PASS" "$CHAR_DB" -e "DELETE FROM group_member WHERE memberGuid NOT IN (SELECT guid FROM characters)" 2>/dev/null

# Delete bot accounts
echo "Deleting bot accounts..."
for acc_id in $BOT_ACCOUNTS; do
    mysql -u "$USER" -p"$PASS" "$AUTH_DB" -e "DELETE FROM account WHERE id = $acc_id" 2>/dev/null
    mysql -u "$USER" -p"$PASS" "$AUTH_DB" -e "DELETE FROM realm_characters WHERE acctid = $acc_id" 2>/dev/null
    mysql -u "$USER" -p"$PASS" "$AUTH_DB" -e "DELETE FROM account_access WHERE id = $acc_id" 2>/dev/null
done

echo "=== Cleanup complete ==="
