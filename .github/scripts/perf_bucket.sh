#!/usr/bin/env bash
# Stores performance benchmark results in the R2 bucket and fetches baselines.
#
# Usage:
#   perf_bucket.sh upload RESULTS
#   perf_bucket.sh baseline RESULTS OUTPUT [REF]
#
# Objects are named {hardware_id}-{commit_sha}-{run_id}.json at the bucket root.
# `baseline` writes to OUTPUT the newest other run on the same hardware for REF's commit
# (a tag, commit or branch) or, without REF, for the newest benchmarked ancestor of the
# commit RESULTS measured.
# It exits 2 when there is no such run.
#
# Requires PERF_BENCH_BUCKET and the AWS_* variables for the bucket.
set -euo pipefail

command=$1
results=$2

hardware_id=$(jq -er '.machine_info.macrosynergy.hardware_id' "$results")
commit_sha=$(jq -er '.machine_info.macrosynergy.commit_sha' "$results")
run_id=$(jq -er '.datetime[:19] | gsub("[-:]"; "") + "Z"' "$results")
key="$hardware_id-$commit_sha-$run_id.json"

case $command in
upload)
    aws s3 cp --only-show-errors "$results" "s3://$PERF_BENCH_BUCKET/$key"
    echo "Uploaded $key"
    ;;
baseline)
    output=$3
    ref=${4:-}
    keys=$(aws s3api list-objects-v2 --bucket "$PERF_BENCH_BUCKET" \
        --prefix "$hardware_id-" --query 'Contents[].Key' --output json |
        jq -r '.[]?' | grep -vFx "$key" || true)
    if [ -n "$ref" ]; then
        commit=$(git rev-parse --verify --quiet "$ref^{commit}" ||
            git rev-parse --verify --quiet "origin/$ref^{commit}" || true)
        if [ -z "$commit" ]; then
            echo "Unknown ref: $ref" >&2
            exit 2
        fi
    else
        commit=$(git rev-list "$commit_sha" | grep -m1 -Fxf <(cut -d- -f2 <<<"$keys") || true)
    fi
    baseline=$(grep "^$hardware_id-${commit:-none}-" <<<"$keys" | sort | tail -n1 || true)
    if [ -z "$baseline" ]; then
        echo "No run on hardware $hardware_id for ${ref:-an ancestor of $commit_sha}" >&2
        exit 2
    fi
    aws s3 cp --only-show-errors "s3://$PERF_BENCH_BUCKET/$baseline" "$output"
    echo "Baseline $baseline"
    ;;
*)
    echo "Unknown command: $command" >&2
    exit 1
    ;;
esac
