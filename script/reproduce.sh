#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
PROFILE="full"
WORKERS=8
SKIP_DOWNLOAD=0

usage() {
  cat <<'USAGE'
Usage: ./script/reproduce.sh <command> [options]

Commands:
  prepare-data       download and convert all six public corpus roots
  smoke              create a tiny corpus and run a four-step smoke training
  train              run the selected pretraining recipe
  run                prepare-data (or smoke) and then train
  test               run the CPU release checks
  benchmark          run the inference benchmark for FP32 and W8
  help               show this message

Options:
  --profile full|local|smoke
                           data/training profile (default: full)
  --workers N            conversion process count (default: 8)
  --skip-download        reuse already downloaded raw data
  PYTHON=<path>          Python interpreter (default: python)
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  printf '\n[%s]\n' "$*"
  "$@"
}

reset_generated_corpus() {
  local path="$1"
  if [[ -e "$path" ]]; then
    printf '[reset] removing incomplete generated corpus only: %s\n' "$path"
    rm -rf "$path"
  fi
}

ensure_corpus() {
  local name="$1"
  local config="$2"
  local path="data/corpus_fast/$name"
  if [[ -f "$path/index.npz" ]]; then
    printf '[skip] %s already has a merged index\n' "$path"
    return
  fi
  reset_generated_corpus "$path"
  run "$PYTHON_BIN" main.py "$config" --workers "$WORKERS"
  run "$PYTHON_BIN" script/corpus/merge_corpus_parts.py --corpus "$path"
}

prepare_full() {
  if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    run "$PYTHON_BIN" script/data/download_public_data.py --all
  fi

  ensure_corpus pret config/corpus/pretrain_full.yaml
  ensure_corpus lotsa config/corpus/lotsa_full.yaml
  ensure_corpus chronos config/corpus/chronos_full.yaml
  ensure_corpus boom config/corpus/boom_full.yaml
  ensure_corpus fev config/corpus/fev_full.yaml

  if [[ ! -f data/corpus_fast/tinycast_synth/index.npz ]]; then
    reset_generated_corpus data/corpus_fast/tinycast_synth
    run "$PYTHON_BIN" script/data/prepare_tinycast_synth.py
  else
    printf '[skip] data/corpus_fast/tinycast_synth already complete\n'
  fi

  run "$PYTHON_BIN" script/corpus/audit_corpus_parts.py --all
  run "$PYTHON_BIN" script/corpus/audit_corpus.py --all
  run "$PYTHON_BIN" script/corpus/audit_corpus_bytes.py --all
}

prepare_smoke() {
  run "$PYTHON_BIN" script/data/make_demo_corpus.py
  run "$PYTHON_BIN" main.py config/corpus/demo.yaml
}

train_selected() {
  case "$PROFILE" in
    full)
      for corpus in pret lotsa chronos boom fev tinycast_synth; do
        [[ -f "data/corpus_fast/$corpus/index.npz" ]] \
          || die "missing data/corpus_fast/$corpus/index.npz; run prepare-data first"
      done
      run "$PYTHON_BIN" main.py config/fracast/pretrain_full.yaml
      ;;
    smoke)
      run "$PYTHON_BIN" main.py config/fracast/pretrain_smoke.yaml
      ;;
    local)
      for corpus in pret lotsa chronos boom fev tinycast_synth; do
        [[ -f "data/corpus_fast/$corpus/index.npz" ]] \
          || die "missing data/corpus_fast/$corpus/index.npz; run prepare-data first"
      done
      run "$PYTHON_BIN" main.py config/fracast/pretrain_local.yaml
      ;;
    *)
      die "unknown profile: $PROFILE (expected full, local, or smoke)"
      ;;
  esac
}

run_tests() {
  run "$PYTHON_BIN" script/tests/test_freq.py
  run "$PYTHON_BIN" script/tests/test_fracast_unit.py
  run "$PYTHON_BIN" script/tests/test_head_future_conv.py
  run "$PYTHON_BIN" script/tests/test_resume_stream.py
  run "$PYTHON_BIN" script/tests/test_fracast_release.py
}

run_benchmark() {
  run "$PYTHON_BIN" script/benchmark_inference.py --weights fp32
  run "$PYTHON_BIN" script/benchmark_inference.py --weights w8
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || die "--profile needs a value"
      PROFILE="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || die "--workers needs a value"
      WORKERS="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "$command" in
  prepare-data)
    [[ "$PROFILE" == "full" ]] || die "prepare-data only supports --profile full"
    prepare_full
    ;;
  smoke)
    PROFILE="smoke"
    prepare_smoke
    train_selected
    ;;
  train)
    train_selected
    ;;
  run)
    if [[ "$PROFILE" == "smoke" ]]; then
      prepare_smoke
      train_selected
    else
      prepare_full
      train_selected
    fi
    ;;
  test)
    run_tests
    ;;
  benchmark)
    run_benchmark
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    die "unknown command: $command"
    ;;
esac
