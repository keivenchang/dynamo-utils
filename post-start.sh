#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs on every container start (postStartCommand).
# GPG private keys, pubring.kbx, and trustdb.gpg are bind-mounted read-only
# from the host. Just ensure ~/.gnupg permissions and restart the agent.

if [ -d "$HOME/.gnupg/private-keys-v1.d" ]; then
    chmod 700 "$HOME/.gnupg" 2>/dev/null
    gpgconf --kill gpg-agent 2>/dev/null
fi
