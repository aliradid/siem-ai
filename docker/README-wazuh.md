# Wazuh setup for the SOAR layer

The paper's SOAR claims are wired through Wazuh Active Response. We run Wazuh
**outside** the main `docker-compose.yml` because the official Wazuh stack
ships its own compose file with specific TLS/cert plumbing.

## 1. Pull the official Wazuh single-node compose

```bash
git clone --depth 1 -b v4.7.5 https://github.com/wazuh/wazuh-docker.git
cd wazuh-docker/single-node
docker compose -f generate-indexer-certs.yml run --rm generator
docker compose up -d
```

This brings up: `wazuh.manager`, `wazuh.indexer`, `wazuh.dashboard`.

Dashboard: <https://localhost:443>  (default creds: `admin / SecretPassword`)

## 2. Connect Wazuh to the SIEM-AI Elasticsearch

In `wazuh-docker/single-node/config/wazuh_cluster/wazuh_manager.conf`, add a
remote forwarder to our stack's Elasticsearch (port 9200 on host network):

```xml
<integration>
  <name>custom-siemai</name>
  <hook_url>http://host.docker.internal:9200/siemai-alerts-*/_doc</hook_url>
  <level>5</level>
  <alert_format>json</alert_format>
</integration>
```

Then restart: `docker compose restart wazuh.manager`.

## 3. Install Active Response scripts on a monitored host

On each host running `wazuh-agent`, drop these into
`/var/ossec/active-response/bin/` and set them executable:

### `quarantine-ip.sh`

```bash
#!/bin/bash
ACTION=$1
IP=$3
case "$ACTION" in
  add)    iptables -I INPUT -s "$IP" -j DROP ;;
  delete) iptables -D INPUT -s "$IP" -j DROP ;;
esac
```

### `disable-account.sh`

```bash
#!/bin/bash
ACTION=$1
USER=$3
# Replace with Azure AD Graph API call or local usermod -L "$USER"
echo "[$(date -Is)] action=$ACTION user=$USER" >> /var/ossec/logs/account-disable.log
```

## 4. Register the responses in `ossec.conf`

```xml
<command>
  <name>quarantine-ip</name>
  <executable>quarantine-ip.sh</executable>
  <timeout_allowed>yes</timeout_allowed>
</command>
<command>
  <name>disable-account</name>
  <executable>disable-account.sh</executable>
  <timeout_allowed>no</timeout_allowed>
</command>

<active-response>
  <command>quarantine-ip</command>
  <location>local</location>
  <rules_id>100200,100201</rules_id>     <!-- our ml_label:malicious rule IDs -->
  <timeout>600</timeout>
</active-response>

<active-response>
  <command>disable-account</command>
  <location>local</location>
  <rules_id>100300</rules_id>
</active-response>
```

## 5. Custom rule to fire on ML verdicts

Append to `/var/ossec/etc/rules/local_rules.xml`:

```xml
<group name="siem-ai,">
  <rule id="100200" level="10">
    <field name="ml_label">malicious</field>
    <description>SIEM-AI: ML model flagged event as malicious</description>
  </rule>
  <rule id="100201" level="12">
    <field name="ml_label">malicious</field>
    <field name="ml_score">^0\.9</field>
    <description>SIEM-AI: high-confidence malicious verdict</description>
  </rule>
</group>
```

Restart the manager: `docker compose restart wazuh.manager`.

## 6. Verify

Tail the active response log:

```bash
docker exec -it single-node-wazuh.manager-1 tail -f /var/ossec/logs/active-responses.log
```

Then run our SOAR validation harness against the live stack:

```bash
cd ../../siem-ai
python scripts/05_soar_validation.py --cycles 50
```

The harness now hits real Wazuh playbooks via the agent's Active Response and
records p50 / p95 / p99 ingest-to-containment latency to `results/soar_validation.json`.
