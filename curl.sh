#!/bin/bash

set -euo pipefail

#
# Convenient script to test a model.
#
# Usage:
#
# ./_curl.sh --model <model> --port <port> --prompt <prompt> --loop
#

show_help() {
  cat << EOF
Usage: $0 [OPTIONS]

A convenient script to test a model via the chat completions API.

OPTIONS:
  --port <port>           Port number for the API server (default: 8080)
  --prompt <prompt>       User prompt to send to the model (default: "Hello!")
  --preprompt <preprompt> System prompt/instruction for the model (default: "You are a helpful assistant.")
  --max-tokens <tokens>   Maximum number of tokens to generate (default: 300)
  --stream                Enable streaming responses (default: false)
  --retry                 Retry the request until it succeeds
  --loop                  Run the test in an infinite loop
  --pause <seconds>       Pause duration between iterations in loop mode (default: 1.0)
  --random                Add a random number prefix to the prompt (only in --loop mode)
  --metrics               Show metrics after the test completes
  --help                  Show this help message

EXAMPLES:
  $0 --port 8080 --prompt "Hello world"
  $0 --port 8080 --prompt "Write a Python function" --preprompt "You are a coding assistant."
  $0 --port 8080 --prompt "Hello world" --stream --retry
  $0 --port 8080 --prompt "Hello world" --loop --metrics

EOF
}

# Validate response has message content (returns 0 for valid, 2 for no content)
validate_response() {
  local response="$1"
  local message_content
  message_content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
  
  if [ -z "$message_content" ] || [ "$message_content" = "null" ] || [ "$message_content" = "empty" ]; then
    echo "NO_MESSAGE_FAILURE: Response received but no message content found" >&2
    echo "Response was: $response" >&2
    return 2
  fi
  return 0
}

# Execute curl and handle errors (returns 1 for curl failure, 2 for validation failure, 0 for success)
execute_curl() {
  local payload="$1"
  local retry_mode="$2"
  local stream="$3"
  
  if [ "$stream" = true ]; then
    printf "<STREAM>"
    curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "$payload" | \
      while IFS= read -r line; do
        if [[ "$line" == data:* ]]; then
          content=$(echo "$line" | sed 's/^data: //' | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
          if [ -n "$content" ] && [ "$content" != "null" ]; then
            printf "%s" "$content"
          fi
        fi
      done
    local curl_exit_code=${PIPESTATUS[0]}
    echo "</STREAM>"
    
    if [ $curl_exit_code -ne 0 ] && [ "$retry_mode" = true ]; then
      echo "CURL_FAILURE: curl command failed with exit code $curl_exit_code" >&2
      return 1
    fi
  else
    local response
    response=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "$payload")
    local curl_exit_code=$?
    
    if [ $curl_exit_code -ne 0 ]; then
      if [ "$retry_mode" = true ]; then
        echo "CURL_FAILURE: curl command failed with exit code $curl_exit_code" >&2
        return 1
      fi
      echo "$response"
      return 1
    fi
    
    # Validate response if in retry mode
    if [ "$retry_mode" = true ]; then
      validate_response "$response" || return $?
    fi
    
    echo "$response"
  fi
  return 0
}

make_curl_request() {
  local model="$1"
  local prompt="$2"
  local preprompt="$3"
  local stream="$4"
  local use_random="$5"
  local max_tokens="$6"
  local retry_mode="${7:-false}"

  # Build the user content
  local user_content="$prompt"
  [ "$use_random" = true ] && user_content="$((RANDOM % 1000)) $prompt"

  # Build payload
  local payload=$(cat <<EOF
{
  "model": "$model",
  "messages": [
    {"role": "developer", "content": "$preprompt"},
    {"role": "user", "content": "$user_content"}
  ],
  "stream": $stream,
  "max_tokens": $max_tokens
}
EOF
)

  echo "$ curl -s localhost:$PORT/v1/chat/completions -H \"Content-Type: application/json\" -d '$payload'"
  execute_curl "$payload" "$retry_mode" "$stream"
}

test_model() {
  local model="$1"
  local prompt="$2"
  local preprompt="$3"
  local stream="$4"
  local retry_until_succ="$5"
  local use_random="$6"
  local max_tokens="$7"

  if [ "$retry_until_succ" = true ]; then
    echo "Retrying until success (with message validation)..."
    local retry_count=0
    
    while ! make_curl_request "$model" "$prompt" "$preprompt" "$stream" "$use_random" "$max_tokens" "$retry_until_succ"; do
      retry_count=$((retry_count + 1))
      case $? in
        1) echo "Retry $retry_count: CURL command failed, retrying in $PAUSE_SEC seconds..." ;;
        2) echo "Retry $retry_count: Response received but no message content, retrying in $PAUSE_SEC seconds..." ;;
        *) echo "Retry $retry_count: Unknown failure, retrying in $PAUSE_SEC seconds..." ;;
      esac
      sleep $PAUSE_SEC
    done
    
    [ $retry_count -gt 0 ] && echo "Success after $retry_count retries!"
  else
    make_curl_request "$model" "$prompt" "$preprompt" "$stream" "$use_random" "$max_tokens" false || 
      echo "Request failed, but not set to retry."
  fi
  echo ""
}

get_metrics() {
    echo "Getting metrics from localhost:$PORT/metrics..."
    curl -s "http://localhost:$PORT/metrics" | grep -v _bucket
}

# Get the default model from the API (retry until found)
get_default_model() {
  #echo "Checking for available models..." >&2
  local retry_count=0
  
  while true; do
    local response=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null)
    local curl_exit_code=$?
    
    if [ $curl_exit_code -eq 0 ]; then
      local model_id=$(echo "$response" | jq -r '.data[0].id' 2>/dev/null)
      if [ -n "$model_id" ] && [ "$model_id" != "null" ] && [ "$model_id" != "empty" ]; then
        [ $retry_count -gt 0 ] && echo "Model found after $retry_count retries: $model_id" >&2
        echo "$model_id"
        return 0
      fi
    fi
    
    retry_count=$((retry_count + 1))
    echo "Retry $retry_count: No models found or API unavailable, retrying in $PAUSE_SEC seconds..." >&2
    sleep $PAUSE_SEC
  done
}

# Default values for command line arguments
PORT=8080
PROMPT="Hello!"
PREPROMPT="You are a helpful assistant."
MAX_TOKENS=300
STREAM=false
RETRY_UNTIL_SUCC=false
SHOW_METRICS=false
LOOP_FOREVER=false
USE_RANDOM=false
PAUSE_SEC=2.0

while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)
      show_help
      exit 0
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --preprompt)
      PREPROMPT="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --stream)
      STREAM=true
      shift
      ;;
    --retry)
      RETRY_UNTIL_SUCC=true
      shift
      ;;
    --loop)
      LOOP_FOREVER=true
      shift
      ;;
    --random)
      USE_RANDOM=true
      shift
      ;;
    --pause)
      PAUSE_SEC="$2"
      shift 2
      ;;
    --metrics)
      SHOW_METRICS=true
      shift
      ;;
    *)
      echo "Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Call test_model with parsed parameters
MODEL=$(get_default_model)

if [ "$LOOP_FOREVER" = true ]; then
  echo "Starting infinite loop of test_model calls..."
  while true; do
    test_model "$MODEL" "$PROMPT" "$PREPROMPT" "$STREAM" "$RETRY_UNTIL_SUCC" "$USE_RANDOM" "$MAX_TOKENS"

    # Show metrics if requested
    if [ "$SHOW_METRICS" = true ]; then
      get_metrics
    fi

    echo "Waiting $PAUSE_SEC seconds before next iteration..."
    sleep $PAUSE_SEC
  done
else
  test_model "$MODEL" "$PROMPT" "$PREPROMPT" "$STREAM" "$RETRY_UNTIL_SUCC" "$USE_RANDOM" "$MAX_TOKENS"

  # Show metrics if requested
  if [ "$SHOW_METRICS" = true ]; then
    get_metrics
  fi
fi
