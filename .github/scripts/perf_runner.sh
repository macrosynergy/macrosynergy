#!/usr/bin/env bash
# Starts and stops the single-use EC2 instance the performance benchmarks run on.
#
# Usage:
#   perf_runner.sh start
#   perf_runner.sh stop RUNNER_ID
#
# `start` registers a just-in-time GitHub runner, launches an instance that runs it (spot,
# then on demand) and, once the runner is online, writes `label` and `runner_id` to
# GITHUB_OUTPUT. If anything fails on the way, it undoes what it started. The instance
# terminates itself when its runner exits: after one job, when `stop` removes the runner,
# or after the configured maximum.
#
# The logs are public, so the script prints no AWS identifiers: it assumes the role itself
# and reports AWS failures by error code only.
#
# The configuration (GitHub App, runner group, instance type, subnets, security group) is
# a JSON SSM parameter named by PERF_RUNNER_CONFIG, managed in aws-infra-management.
# In GitHub Actions it assumes the role in PERF_RUNNER_ROLE; elsewhere it uses the
# credentials it finds.
set -euo pipefail
shopt -s inherit_errexit

readonly UBUNTU_IMAGE=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
readonly ONLINE_ATTEMPTS=60 # 10 s apart

org=$GITHUB_REPOSITORY_OWNER
work=$(mktemp -d)
umask 077

instance_id=""
runner_id=""
undo_on_failure=false

finish() {
    local status=$?
    if [ "$status" -ne 0 ] && $undo_on_failure; then
        terminate_instance "$instance_id" || true
        remove_leftover_runner "$runner_id"
    fi
    rm -rf "$work"
    exit "$status"
}
trap finish EXIT

# Configuration --------------------------------------------------------------------------

load_config() {
    aws_cli ssm get-parameter --name "$PERF_RUNNER_CONFIG" \
        --query Parameter.Value --output text > "$work/config.json"
}

config() {
    jq -er "$1" "$work/config.json"
}

# GitHub ---------------------------------------------------------------------------------

base64url() {
    openssl base64 -A | tr '+/' '-_' | tr -d '='
}

# Calls the GitHub API with the Authorization header in $work/auth and prints the body.
github() {
    local method=$1 path=$2 body=${3:-} status
    status=$(curl -sS -o "$work/response" -w '%{http_code}' -X "$method" \
        -H @"$work/auth" -H 'Accept: application/vnd.github+json' \
        ${body:+--data "$body"} "https://api.github.com$path")
    if [[ $status != 2* ]]; then
        echo "GitHub answered $status to $method $path: $(jq -r '.message?' "$work/response")" >&2
        return 1
    fi
    cat "$work/response"
}

# Signs in as the GitHub App's installation. The key and tokens only ever sit in files.
sign_in() {
    local now claims jwt
    aws_cli secretsmanager get-secret-value --secret-id "$(config .key_secret)" \
        --query SecretString --output text > "$work/key.pem"
    now=$(date +%s)
    claims=$(jq -nc --arg app "$(config .app_id)" --argjson now "$now" \
        '{iat: ($now - 60), exp: ($now + 540), iss: $app}')
    jwt="$(printf '{"alg":"RS256"}' | base64url).$(printf '%s' "$claims" | base64url)"
    jwt="$jwt.$(printf '%s' "$jwt" | openssl dgst -sha256 -sign "$work/key.pem" -binary | base64url)"
    printf 'Authorization: Bearer %s\n' "$jwt" > "$work/auth"

    github POST "/app/installations/$(config .installation_id)/access_tokens" |
        jq -r '"Authorization: Bearer \(.token)"' > "$work/auth.next"
    mv "$work/auth.next" "$work/auth"
}

runner_group_id() {
    github GET "/orgs/$org/actions/runner-groups" |
        jq -er --arg name "$(config .runner_group)" '.runner_groups[] | select(.name == $name) | .id'
}

# Registers a runner that takes one job labelled $1. Leaves its JIT config in $work/jit.
register_runner() {
    local label=$1 request
    request=$(jq -nc --arg label "$label" --argjson group "$(runner_group_id)" \
        '{name: $label, runner_group_id: $group, labels: [$label]}')
    github POST "/orgs/$org/actions/runners/generate-jitconfig" "$request" > "$work/jit"
    jq -r '.runner.id' "$work/jit"
}

latest_runner_version() {
    github GET /repos/actions/runner/releases/latest | jq -er '.tag_name | ltrimstr("v")'
}

wait_until_online() {
    local runner=$1 status=""
    for _ in $(seq "$ONLINE_ATTEMPTS"); do
        status=$(github GET "/orgs/$org/actions/runners/$runner" | jq -r '.status') || status=unreachable
        [ "$status" = online ] && return 0
        sleep 10
    done
    echo "Runner $runner is still $status after $((ONLINE_ATTEMPTS * 10)) s" >&2
    return 1
}

# A runner that took its job has removed itself, so a failure here is expected.
remove_leftover_runner() {
    if [ -n "$1" ] && github DELETE "/orgs/$org/actions/runners/$1" > /dev/null 2>&1; then
        echo "Removed runner $1"
    fi
}

# AWS ------------------------------------------------------------------------------------

# Runs the AWS CLI. On failure it prints only the error code: AWS's messages name the
# account and the role.
aws_cli() {
    local status=0 code
    aws "$@" 2> "$work/aws-error" || status=$?
    if [ "$status" -ne 0 ]; then
        code=$(grep -oE "\\([A-Za-z]+\\)" "$work/aws-error" | head -n1 | tr -d "()") || true
        echo "AWS $1 $2 failed: ${code:-exit $status}" >&2
    fi
    return "$status"
}

# In GitHub Actions, assumes PERF_RUNNER_ROLE with the job's OIDC token. Done here rather
# than with aws-actions/configure-aws-credentials, which prints the role's ID.
assume_role() {
    local value
    [ -n "${ACTIONS_ID_TOKEN_REQUEST_URL:-}" ] || return 0
    printf "Authorization: bearer %s\n" "$ACTIONS_ID_TOKEN_REQUEST_TOKEN" > "$work/oidc.h"
    curl -sSf -H @"$work/oidc.h" "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=sts.amazonaws.com" |
        jq -r .value > "$work/oidc-token"
    aws_cli sts assume-role-with-web-identity --role-arn "$PERF_RUNNER_ROLE" \
        --role-session-name "perf-$GITHUB_RUN_ID" --web-identity-token "file://$work/oidc-token" \
        --query Credentials --output json > "$work/credentials"
    AWS_ACCESS_KEY_ID=$(jq -r .AccessKeyId "$work/credentials")
    AWS_SECRET_ACCESS_KEY=$(jq -r .SecretAccessKey "$work/credentials")
    AWS_SESSION_TOKEN=$(jq -r .SessionToken "$work/credentials")
    for value in "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$AWS_SESSION_TOKEN"; do
        echo "::add-mask::$value"
    done
    export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
}

# The instance's user data: installs the runner, runs one job, then shuts down. Python
# goes in /opt/hostedtoolcache, the only place actions/setup-python's Linux builds run from.
write_user_data() {
    local version=$1
    cat > "$work/user-data" <<EOF
#!/bin/bash
set -euo pipefail
trap 'shutdown -h now' EXIT
shutdown -h +$(config .max_minutes)

echo 'DPkg::Lock::Timeout "600";' > /etc/apt/apt.conf.d/90-wait-for-lock
mkdir -p /home/ubuntu/runner /opt/hostedtoolcache
cd /home/ubuntu/runner
curl -fsSL https://github.com/actions/runner/releases/download/v$version/actions-runner-linux-x64-$version.tar.gz | tar xz
./bin/installdependencies.sh
chown -R ubuntu: /home/ubuntu/runner /opt/hostedtoolcache

sudo -u ubuntu env AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache \
    ./run.sh --jitconfig $(jq -r '.encoded_jit_config' "$work/jit")
EOF
}

# Launches image $4 as one instance in $2 ("spot" or "on-demand") and subnet $3, tagged
# with label $1.
run_instance() {
    local label=$1 market=$2 subnet=$3 image=$4 options=()
    if [ "$market" = spot ]; then
        options+=(--instance-market-options
            'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
    fi
    aws_cli ec2 run-instances "${options[@]}" \
        --image-id "$image" \
        --instance-type "$(config .instance_type)" \
        --subnet-id "$subnet" \
        --security-group-ids "$(config .security_group)" \
        --metadata-options HttpTokens=required \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
        --instance-initiated-shutdown-behavior terminate \
        --user-data "file://$work/user-data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$label},{Key=msy:perf-runner,Value=true}]" \
        --query 'Instances[0].InstanceId' --output text
}

# Tries spot in every subnet, then on demand in every subnet.
launch_instance() {
    local label=$1 image market subnet count index
    count=$(config ".subnets | length")
    image=$(aws_cli ssm get-parameter --name "$UBUNTU_IMAGE" --query Parameter.Value --output text)
    for market in spot on-demand; do
        for index in $(seq 0 $((count - 1))); do
            subnet=$(config ".subnets[$index]")
            if run_instance "$label" "$market" "$subnet" "$image"; then
                return 0
            fi
            echo "Could not launch $market in subnet $((index + 1)) of $count" >&2
        done
    done
    echo "No instance could be launched" >&2
    return 1
}

terminate_instance() {
    if [ -n "$1" ]; then
        aws_cli ec2 terminate-instances --instance-ids "$1" > /dev/null
        echo "Terminated the instance"
    fi
}

# Commands -------------------------------------------------------------------------------

start() {
    local label="perf-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" version
    undo_on_failure=true
    assume_role
    load_config
    sign_in

    runner_id=$(register_runner "$label")
    version=$(latest_runner_version)
    write_user_data "$version"
    instance_id=$(launch_instance "$label")
    echo "Launched an instance for runner $label"

    wait_until_online "$runner_id"
    printf 'label=%s\nrunner_id=%s\n' "$label" "$runner_id" >> "$GITHUB_OUTPUT"
}

# Removes the runner; its instance then shuts itself down.
stop() {
    assume_role
    load_config
    sign_in
    remove_leftover_runner "$1"
}

case ${1:-} in
start) start ;;
stop) stop "${2:-}" ;;
*)
    echo "Usage: perf_runner.sh start | stop RUNNER_ID" >&2
    exit 1
    ;;
esac
