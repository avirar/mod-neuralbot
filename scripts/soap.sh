#!/usr/bin/env bash
# Send a command to the worldserver console over SOAP (port 7878).
# Usage: scripts/soap.sh "server info"
# Uses the PI admin account (gmlevel 3) — see AGENTS.md "SOAP console access".
set -u
CMD="${1:?command required}"
export NEURALBOT_SOAP_USER="${NEURALBOT_SOAP_USER:-PI}"
export NEURALBOT_SOAP_PASS="${NEURALBOT_SOAP_PASS:-piagent2026}"
export NEURALBOT_SOAP_URL="${NEURALBOT_SOAP_URL:-http://127.0.0.1:7878/}"

python3 - "$CMD" <<'PY'
import sys, os, urllib.request, base64, re
cmd = sys.argv[1]
url = os.environ["NEURALBOT_SOAP_URL"]
auth = base64.b64encode(f"{os.environ['NEURALBOT_SOAP_USER']}:{os.environ['NEURALBOT_SOAP_PASS']}".encode()).decode()
body = ('<?xml version="1.0"?><SOAP-ENV:Envelope '
        'xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:ns1="urn:AC">'
        '<SOAP-ENV:Body><ns1:executeCommand>'
        f'<command>{cmd}</command>'
        '</ns1:executeCommand></SOAP-ENV:Body></SOAP-ENV:Envelope>').encode()
req = urllib.request.Request(url, method="POST", data=body)
req.add_header("Content-Type", "text/xml")
req.add_header("Authorization", "Basic " + auth)
resp = urllib.request.urlopen(req, timeout=30).read().decode()
m = re.search(r"<result>(.*?)</result>", resp, re.DOTALL)
if m:
    print(m.group(1).replace("&#xD;", "\n"))
else:
    print("(no result)", file=sys.stderr)
PY
