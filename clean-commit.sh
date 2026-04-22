#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scrub the current HEAD commit message:
#   1. Remove lines starting with "Made-with:"
#   2. Warn loudly and redact ALL @nvidia.com email addresses from the
#      entire message (body + trailers). Addresses inside <...> become <>;
#      bare addresses become <redacted>.
#   3. Drop Signed-off-by lines whose email is empty (e.g. "Name <>")
#      — these are left behind by redaction and add no value.
#   4. De-duplicate "Signed-off-by:" lines (keep first occurrence of each)
#   5. Remove "Co-Authored-By:" (or "Co-authored-by:") lines that duplicate
#      a Signed-off-by trailer (same name + email). Non-duplicate co-authors
#      are preserved.
#   6. Strip trailing blank lines
#
# Usage:
#   clean-commit.sh              # clean HEAD (amend)
#   clean-commit.sh --dry-run    # show original + cleaned, don't amend
#   clean-commit.sh --self-test  # run built-in test cases

set -euo pipefail

# ---------------------------------------------------------------------------
# Core transforms (pure functions — read stdin, write stdout / stderr)
# ---------------------------------------------------------------------------

# Emit a loud stderr warning if the message contains any @nvidia.com address.
# Reads the message on stdin; does not modify it. Always returns 0.
nvidia_warning() {
    local msg hits
    msg=$(cat)
    hits=$(printf '%s\n' "$msg" | grep -oE '[A-Za-z0-9._%+-]+@nvidia\.com' | sort -u || true)
    if [[ -z "$hits" ]]; then
        return 0
    fi
    {
        echo
        echo "########################################################################"
        echo "# WARNING: @nvidia.com email address(es) found in commit message:"
        while IFS= read -r addr; do
            echo "#   - $addr"
        done <<<"$hits"
        echo "#"
        echo "# NEVER add @nvidia.com addresses to commit messages. Use your public"
        echo "# identity (e.g. <name@users.noreply.github.com>) instead. This script"
        echo "# will redact them now, but fix your git config to prevent it:"
        echo "#   git config --global user.email <your-public-email>"
        echo "########################################################################"
        echo
    } >&2
}

# Transform a commit message (stdin -> stdout). See header comment for rules.
clean_message() {
    awk '
        function trailer_ident(line,    s) {
            # Strip the trailer prefix (everything up to and including the
            # first ":") and surrounding whitespace so "Signed-off-by: X <e>"
            # and "Co-authored-by: X <e>" compare equal when name/email match.
            s = line
            sub(/^[^:]+:[[:space:]]*/, "", s)
            return s
        }
        {
            if ($0 ~ /^Made-with:/) next
            # Redact any @nvidia.com address anywhere in the line. Handle
            # both wrapped (<name@nvidia.com>) and bare (name@nvidia.com).
            gsub(/<[^>]*@nvidia\.com>/, "<>")
            gsub(/[A-Za-z0-9._%+-]+@nvidia\.com/, "<redacted>")
            if ($0 ~ /^Signed-off-by:/) {
                # Drop signoffs left with an empty email (e.g. "Name <>")
                # after redaction — they add no value.
                if ($0 ~ /<>[[:space:]]*$/) next
                ident = trailer_ident($0)
                if (ident in signoff_seen) next
                signoff_seen[ident] = 1
            }
            lines[++n] = $0
        }
        END {
            # Filter Co-authored-by lines that duplicate a Signed-off-by.
            m = 0
            for (i = 1; i <= n; i++) {
                if (lines[i] ~ /^[Cc]o-[Aa]uthored-[Bb]y:/) {
                    if (trailer_ident(lines[i]) in signoff_seen) continue
                }
                out[++m] = lines[i]
            }
            while (m > 0 && out[m] ~ /^[[:space:]]*$/) m--
            for (i = 1; i <= m; i++) print out[i]
        }
    '
}

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_test_result=0

_check() {
    # Args: name  input  expected_output
    local name="$1" input="$2" expected="$3" actual
    actual=$(printf '%s\n' "$input" | clean_message)
    if [[ "$actual" == "$expected" ]]; then
        printf '  OK    %s\n' "$name"
    else
        printf '  FAIL  %s\n' "$name"
        printf '    --- expected ---\n'
        printf '%s\n' "$expected" | sed 's/^/    /'
        printf '    --- actual ---\n'
        printf '%s\n' "$actual" | sed 's/^/    /'
        _test_result=1
    fi
}

_check_warning() {
    # Args: name  input  should_warn(yes|no)
    local name="$1" input="$2" should_warn="$3" stderr
    stderr=$(printf '%s\n' "$input" | nvidia_warning 2>&1 >/dev/null)
    local saw_warning=no
    [[ "$stderr" == *"WARNING: @nvidia.com"* ]] && saw_warning=yes
    if [[ "$saw_warning" == "$should_warn" ]]; then
        printf '  OK    %s\n' "$name"
    else
        printf '  FAIL  %s  (expected warn=%s, got warn=%s)\n' \
            "$name" "$should_warn" "$saw_warning"
        _test_result=1
    fi
}

self_test() {
    echo "=== clean_message ==="

    _check "drop empty-email signoff" \
"fix: something

Signed-off-by: Keiven Chang <keivenc@nvidia.com>" \
"fix: something"

    _check "drop dup co-author" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>
Co-authored-by: Keiven Chang <keivenchang@users.noreply.github.com>" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>"

    _check "keep non-dup co-author" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>
Co-authored-by: Someone Else <someone@example.com>" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>
Co-authored-by: Someone Else <someone@example.com>"

    _check "redact nvidia co-author, preserved when no matching signoff" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>
Co-authored-by: Keiven Chang <keivenc@nvidia.com>" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>
Co-authored-by: Keiven Chang <>"

    _check "redact nvidia in body (bare address)" \
"fix: something

contact me at keivenc@nvidia.com for details

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>" \
"fix: something

contact me at <redacted> for details

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>"

    _check "no nvidia = untouched" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>"

    _check "strip Made-with" \
"fix: something

Made-with: magic
Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>"

    _check "dedup signoffs + drop empty + strip matching co-author" \
"fix: something

Signed-off-by: Keiven Chang <keivenc@nvidia.com>
Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>
Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>
Co-authored-by: Keiven Chang <keivenchang@users.noreply.github.com>" \
"fix: something

Signed-off-by: Keiven Chang <keivenchang@users.noreply.github.com>"

    _check "trim trailing blank lines" \
"fix: something

body line

" \
"fix: something

body line"

    echo
    echo "=== nvidia_warning ==="

    _check_warning "warn on nvidia in signoff" \
"fix: bad

Signed-off-by: X <keivenc@nvidia.com>" yes

    _check_warning "warn on nvidia in body" \
"fix: bad

contact keivenc@nvidia.com" yes

    _check_warning "no warn when clean" \
"fix: ok

Signed-off-by: X <x@users.noreply.github.com>" no

    _check_warning "no warn on similar but non-nvidia domain" \
"fix: ok

email: me@not-nvidia.com" no

    echo
    if [[ "$_test_result" -eq 0 ]]; then
        echo "ALL TESTS PASSED"
    else
        echo "SOME TESTS FAILED"
    fi
    return "$_test_result"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DRY_RUN=0
case "${1:-}" in
    --self-test)
        self_test
        exit $?
        ;;
    --dry-run)
        DRY_RUN=1
        ;;
    "")
        ;;
    *)
        echo "Usage: $0 [--dry-run | --self-test]" >&2
        exit 2
        ;;
esac

original=$(git log -1 --format="%B")

printf '%s\n' "$original" | nvidia_warning

cleaned=$(printf '%s\n' "$original" | clean_message)

if [[ "$original" == "$cleaned"$'\n' ]] || [[ "$original" == "$cleaned" ]]; then
    echo "Nothing to clean."
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--- original ---"
    printf '%s\n' "$original"
    echo "--- cleaned ---"
    printf '%s\n' "$cleaned"
    exit 0
fi

GIT_EDITOR=true git commit --amend -m "$cleaned"
echo "Commit message cleaned."
