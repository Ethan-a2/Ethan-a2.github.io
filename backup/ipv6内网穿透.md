
# 步骤
1. 光猫改桥接。
2. 路由器使能ipv6。
3. 域名托管到cloudflare，可使用免费注册的dpdns.org域名。
4. 获取cloudflare API token。
5. cloudflare中增加一条AAAA记录，Name填: ddnsv6，Content填写: 240e:341:: , 
6. 在内网ubuntu主机上配置DDNS脚本: cloudflare_ddns_ipv6.sh，修改其中的CF_API_TOKEN、CF_ZONE_ID、RECORD_NAME。
	1. CF_API_TOKEN为第4步中申请的token。
	2. CF_ZONE_ID在cloudflare的主机中，点击xxx.dpdns.org域名，进入页面的右下角查看。
	3. RECORD_NAME为第5步中添加的AAAA记录，比如ddnsv6.xxx..dpdns.org。
7. crontab -l定时执行脚本
	```
	*/15 * * * * /usr/local/bin/cloudflare_ddns_ipv6.sh
	```
8. cloudflare_ddns_ipv6.sh内容如下:
	```
	#!/usr/bin/env bash
	
	# === Configuration ===
	# --- Cloudflare API ---
	# Create a Cloudflare API Token with Zone:DNS:Edit permissions for your specific zone
	CF_API_TOKEN="YOUR_CLOUDFLARE_API_TOKEN" # Replace with your token
	CF_ZONE_ID="YOUR_CLOUDFLARE_ZONE_ID"     # Replace with your Zone ID
	# --- DNS Record ---
	RECORD_NAME="your_subdomain.yourdomain.com" # Replace with the FQDN to update (e.g., home.example.com)
	RECORD_TYPE="AAAA"                          # Use "A" for IPv4, "AAAA" for IPv6
	# --- Behaviour ---
	CF_PROXIED="false"                          # Use "true" for proxied (Orange Cloud), "false" for DNS only (Grey Cloud)
	# --- State & Logging ---
	# Optional: Use absolute paths if running from cron
	STATE_FILE="/tmp/cf_ddns_ipv6.lastip"       # Stores the last known public IP address
	LOG_FILE="/tmp/cloudflare_ddns_ipv6.log" # Log file location
	# --- External IP Service ---
	# Service that returns *only* your public IPv6 address
	IPV6_SERVICE="https://api64.ipify.org"      # Alternatives: ip6.icanhazip.com, ifconfig.me/ip (sometimes returns v4/v6)
	# === End Configuration ===
	
	# === Helper Functions ===
	log() {
	    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
	    # Optionally echo to stdout as well if running interactively
	    # echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
	}
	
	# Basic IPv6 Address Regex (adjust if needed for specific formats like ::1)
	# This regex is simplified and checks for common structures.
	is_ipv6() {
	    [[ "$1" =~ ^([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}$|^([0-9a-fA-F]{1,4}:){1,7}:$|^([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}$|^([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}$|^([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}$|^([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}$|^([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}$|^[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})$|^:((:[0-9a-fA-F]{1,4}){1,7}|:)$|^fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}$|^::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])$|^([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])$ ]]
	}
	
	# === Sanity Checks ===
	# Check if required tools exist
	if ! command -v curl &> /dev/null; then
	    echo "Error: curl is not installed." >&2
	    exit 1
	fi
	if ! command -v jq &> /dev/null; then
	    echo "Error: jq is not installed." >&2
	    exit 1
	fi
	
	# Check if configuration seems valid (basic checks)
	if [[ "$CF_API_TOKEN" == "YOUR_CLOUDFLARE_API_TOKEN" ]] || [[ -z "$CF_API_TOKEN" ]]; then
	    echo "Error: CF_API_TOKEN is not configured." >&2; exit 1; fi
	if [[ "$CF_ZONE_ID" == "YOUR_CLOUDFLARE_ZONE_ID" ]] || [[ -z "$CF_ZONE_ID" ]]; then
	    echo "Error: CF_ZONE_ID is not configured." >&2; exit 1; fi
	if [[ "$RECORD_NAME" == "your_subdomain.yourdomain.com" ]] || [[ -z "$RECORD_NAME" ]]; then
	    echo "Error: RECORD_NAME is not configured." >&2; exit 1; fi
	if [[ "$RECORD_TYPE" != "AAAA" ]]; then
	    echo "Error: This script is configured for AAAA records, but RECORD_TYPE is set to '$RECORD_TYPE'." >&2; exit 1; fi
	
	# Create log file directory if it doesn't exist
	LOG_DIR=$(dirname "$LOG_FILE")
	if [[ ! -d "$LOG_DIR" ]]; then
	    mkdir -p "$LOG_DIR" || { echo "Error: Could not create log directory '$LOG_DIR'." >&2; exit 1; }
	fi
	# Ensure log file is writable
	touch "$LOG_FILE" || { echo "Error: Could not write to log file '$LOG_FILE'." >&2; exit 1; }
	
	# === Main Logic ===
	
	log "Starting DDNS update check for ${RECORD_NAME}..."
	
	# --- Get Current Public IPv6 Address ---
	CURRENT_IPV6=$(curl -s -6 "$IPV6_SERVICE") # Use -6 to force IPv6 request
	if [[ $? -ne 0 ]]; then
	    log "Error: Failed to contact IPv6 service '$IPV6_SERVICE'."
	    exit 1
	fi
	
	# Validate the fetched IP
	if ! is_ipv6 "$CURRENT_IPV6"; then
	    # Sometimes services return error messages or IPv4 if IPv6 isn't available
	    log "Error: Fetched IP '$CURRENT_IPV6' from '$IPV6_SERVICE' does not look like a valid IPv6 address."
	    exit 1
	fi
	log "Current public IPv6: ${CURRENT_IPV6}"
	
	# --- Get Last Known IP Address ---
	LAST_IPV6=""
	if [[ -f "$STATE_FILE" ]]; then
	    LAST_IPV6=$(cat "$STATE_FILE")
	    log "Last known IP from state file: ${LAST_IPV6}"
	else
	    log "State file '${STATE_FILE}' not found. Will force update."
	fi
	
	# --- Compare IP Addresses ---
	if [[ "$CURRENT_IPV6" == "$LAST_IPV6" ]]; then
	    log "IP address unchanged (${CURRENT_IPV6}). No update needed."
	    exit 0
	fi
	
	log "IP address changed (Current: ${CURRENT_IPV6}, Previous: ${LAST_IPV6}). Updating Cloudflare..."
	
	# --- Cloudflare API Interaction ---
	
	# Common Headers
	HEADERS=(-H "Authorization: Bearer ${CF_API_TOKEN}" -H "Content-Type: application/json")
	
	# --- Get Record ID ---
	API_URL_GET="https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/dns_records?type=${RECORD_TYPE}&name=${RECORD_NAME}"
	log "Fetching Record ID from Cloudflare for ${RECORD_NAME}..."
	
	RECORD_INFO_JSON=$(curl -s -X GET "$API_URL_GET" "${HEADERS[@]}")
	if [[ $? -ne 0 ]]; then
	    log "Error: Failed to connect to Cloudflare API (GET)."
	    exit 1
	fi
	
	# Check if the API call was successful before parsing
	SUCCESS=$(echo "$RECORD_INFO_JSON" | jq -r '.success')
	if [[ "$SUCCESS" != "true" ]]; then
	    ERRORS=$(echo "$RECORD_INFO_JSON" | jq -r '.errors | map("\(.code) \(.message)") | join(", ")')
	    log "Error: Cloudflare API (GET) failed. Errors: ${ERRORS:-Unknown API error, check JSON response}"
	    log "API Response (GET): ${RECORD_INFO_JSON}"
	    exit 1
	fi
	
	# Extract Record ID
	RECORD_ID=$(echo "$RECORD_INFO_JSON" | jq -r '.result[0].id') # Get ID of the first matching record
	
	if [[ -z "$RECORD_ID" ]] || [[ "$RECORD_ID" == "null" ]]; then
	    log "Error: Could not find Record ID for ${RECORD_NAME} (Type: ${RECORD_TYPE}) in zone ${CF_ZONE_ID}."
	    log "API Response (GET): ${RECORD_INFO_JSON}"
	    log "Check if the record exists in your Cloudflare DNS settings."
	    # Optional: Create the record if it doesn't exist (more complex, involves POST request)
	    # log "Attempting to create the record..."
	    exit 1
	fi
	
	log "Found Record ID: ${RECORD_ID}"
	
	# --- Update DNS Record ---
	API_URL_PUT="https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/dns_records/${RECORD_ID}"
	log "Updating DNS record ${RECORD_ID} for ${RECORD_NAME} to ${CURRENT_IPV6}..."
	
	# Construct JSON payload using jq for safety with special characters
	JSON_PAYLOAD=$(jq -n \
	    --arg type "$RECORD_TYPE" \
	    --arg name "$RECORD_NAME" \
	    --arg content "$CURRENT_IPV6" \
	    --argjson proxied "$CF_PROXIED" \
	    '{"type": $type, "name": $name, "content": $content, "ttl": 1, "proxied": $proxied}') # ttl:1 = Auto
	
	UPDATE_RESPONSE_JSON=$(curl -s -X PUT "$API_URL_PUT" "${HEADERS[@]}" --data "$JSON_PAYLOAD")
	if [[ $? -ne 0 ]]; then
	    log "Error: Failed to connect to Cloudflare API (PUT)."
	    exit 1
	fi
	
	# Check if the update was successful
	SUCCESS=$(echo "$UPDATE_RESPONSE_JSON" | jq -r '.success')
	if [[ "$SUCCESS" == "true" ]]; then
	    log "Success: Cloudflare DNS record updated successfully for ${RECORD_NAME} to ${CURRENT_IPV6}."
	    # --- Update State File ---
	    echo "$CURRENT_IPV6" > "$STATE_FILE"
	    if [[ $? -ne 0 ]]; then
	      log "Warning: Failed to update state file '${STATE_FILE}'."
	    else
	      log "State file '${STATE_FILE}' updated."
	    fi
	    exit 0
	else
	    ERRORS=$(echo "$UPDATE_RESPONSE_JSON" | jq -r '.errors | map("\(.code) \(.message)") | join(", ")')
	    log "Error: Cloudflare API (PUT) failed. Errors: ${ERRORS:-Unknown API error, check JSON response}"
	    log "API Response (PUT): ${UPDATE_RESPONSE_JSON}"
	    exit 1
	fi
	
	```
9. 通过如命令查看日志，每15分钟会配置一次。
	```
	cat  /tmp/cloudflare_ddns_ipv6.log 
	```
10. 



# 验证网站
test-ipv6.com


# 其它可尝试方案
- ddns-go取代cloudflare_ddns_ipv6.sh。

# Reference
- https://blog.csdn.net/luo_fengyuan/article/details/135940120
- [一套免费且实用的内网穿透方案！DDNS+Cloudflare给你全新体验。_免费ddns-CSDN博客](https://blog.csdn.net/luo_fengyuan/article/details/135940120)
- [免费域名能干什么？CDN，个人网站，访问家庭内网等_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1uAVLzCEmH/?spm_id_from=333.337.search-card.all.click)
