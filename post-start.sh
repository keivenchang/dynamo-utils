#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs on every container start (postStartCommand).
# Copies GPG key material from the read-only host mount into a container-local
# writable ~/.gnupg so signing works without risking the host keyring.

GNUPG_HOST="$HOME/.gnupg-host"
GNUPG_LOCAL="$HOME/.gnupg"

if [ -d "$GNUPG_HOST/private-keys-v1.d" ]; then
    mkdir -p "$GNUPG_LOCAL/private-keys-v1.d"
    chmod 700 "$GNUPG_LOCAL" "$GNUPG_LOCAL/private-keys-v1.d"

    cp "$GNUPG_HOST"/private-keys-v1.d/*.key "$GNUPG_LOCAL/private-keys-v1.d/" 2>/dev/null
    cp "$GNUPG_HOST/pubring.kbx" "$GNUPG_LOCAL/" 2>/dev/null
    cp "$GNUPG_HOST/trustdb.gpg" "$GNUPG_LOCAL/" 2>/dev/null

    gpgconf --kill gpg-agent 2>/dev/null
fi
