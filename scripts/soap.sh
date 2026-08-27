#!/usr/bin/env bash
# Send a command to the worldserver console over SOAP (port 7878).
# Usage: scripts/soap.sh "server info"
# Uses the PI admin account (gmlevel 3) — see AGENTS.md "SOAP console access".
set -u
SOAP_USER="${NEURALBOT_SOAP_USER:-PI}"
SOAP_PASS="${NEURALBOT_SOAP_PASS:-piagent2026}"
SOAP_URL="${NEURALBOT_SOAP_URL:-http://127.0.0.1:7878/}"
CMD="${1:?command required}"

curl -s --max-time 30 -u "$SOAP_USER:$SOAP_PASS" -H "Content-Type: text/xml" \
  -d "<?xml version=\"1.0\"?><SOAP-ENV:Envelope xmlns:SOAP-ENV=\"http://schemas.xmlsoap.org/soap/envelope/\" xmlns:ns1=\"urn:AC\"><SOAP-ENV:Body><ns1:executeCommand><command>$CMD</command></ns1:executeCommand></SOAP-ENV:Body></SOAP-ENV:Envelope>" \
  "$SOAP_URL" \
  | grep -oE '<result>.*</result>' | sed 's/<result>//;s/<\/result>//' | sed 's/&#xD;/\n/g'
