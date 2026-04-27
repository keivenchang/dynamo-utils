<!-- Source: https://gitlab-master.nvidia.com/keivenc/ai-config/-/blob/master/claude/skills/dyn-pull-latest-image/SKILL.md
     Mirrored here for downloadable access from the commit history dashboard.
     The GitLab copy is the source of truth. -->
---
name: dyn-pull-latest-image
user-invocable: true
description: >-
  Find and pull a Dynamo Docker image (dev or local-dev) by querying ACR /
  ECR directly — either the absolute latest, or the image that matches
  (or is closest to) the user's current local repo SHA. No JSON index
  required; registry tags + git log are the only inputs. Knows the
  root vs UID-remapped (dev vs local-dev) distinction, runs token
  rotation itself, defers only to one-time browser auth.

  Triggers: "pull the latest image", "get a vllm dev image", "pull a
  local-dev image", "image for HEAD", "image for sha XXXX", "match my
  branch".
---

# Pull Dynamo Docker Image (dev / local-dev)

End-to-end: authenticate → fetch index → optionally locate the image
matching the user's local SHA (or fall back to latest) → emit pull
command (dev or local-dev) → explain trade-offs.

## Prerequisites

You need the registry CLIs installed and authed once (see Authentication
section). Hostnames and account IDs are inlined in this skill — no env
file sourcing required.

## Dev vs local-dev — what the user is actually choosing

Both come from the same upstream Docker build. The critical difference
is **who the container runs as**:

| Image | Runs as | Risk / behavior with bind-mounts |
|---|---|---|
| **dev** (older) | `root` inside the container | ⚠️ Anything the container writes to a bind-mounted host directory ends up **root-owned on your host FS**. You then have to `sudo chown` to clean up. Fine for ephemeral / CI-style runs that don't write back. |
| **local-dev** (newer, recommended) | Your regular host user (UID/GID remapped to match `id -u` / `id -g`) | Bind-mounted writes land with your normal ownership — no root-owned droppings in your home dir. This is what you almost always want for editing source. |

`build_localdev_from_dev.py` is what does the remap — it takes a dev image
and layers in a `USER` matching the host. Script lives at
`http://speedoflight.nvidia.com/dynamo/dynamo-utils/container/build_localdev_from_dev.py`
(no local checkout required — the `local_dev_cmd` field in the index already
fetches it inline).

**Default recommendation: local-dev** unless the user explicitly says they
want dev (e.g. reproducing a CI environment exactly, or a one-shot run with
no bind-mount).

## Authentication

The skill must remind the user to authenticate before pulling. Pick the
registry the chosen image lives in (the `pull_cmd` URL tells you which).

Once the underlying credential is set up (one-time browser flow per
machine), Claude can run the **token-rotation commands directly** via
the Bash tool — no need to invoke a wrapper script.

| Registry | Token-rotation command (Claude runs directly) | TTL |
|---|---|---|
| **AWS ECR** (`210086341041.dkr.ecr.us-west-2.amazonaws.com`) | `aws ecr get-login-password --region us-west-2 \| docker login --username AWS --password-stdin 210086341041.dkr.ecr.us-west-2.amazonaws.com` | ~12 h |
| **Azure ACR** (`dynamoci.azurecr.io`) | `az acr login --name dynamoci` | ~3 h |
| **GitLab** (`gitlab-master.nvidia.com:5005`) | `docker login -u <user> -p $(cat ~/.config/gitlab-token) gitlab-master.nvidia.com:5005` | persistent |

**First-time setup (cannot be automated — requires browser):** if
`aws ecr get-login-password` errors with "Unable to locate credentials" or
`az acr login` errors with "Please run 'az login'", Claude must defer to
the user to run the wrapper script once:

```
ECR  →  curl -fsSL http://speedoflight.nvidia.com/dynamo/dynamo-utils/nvidia-aws-ecr-login.sh | bash
ACR  →  curl -fsSL http://speedoflight.nvidia.com/dynamo/dynamo-utils/nvidia-az-acr-login.sh | bash
```

These scripts walk the user through nvsec (ECR) / `az login` (ACR), which
require an interactive browser. After that one-time setup, the rotation
commands above work non-interactively for the lifetime of the underlying
credential.

Default registry preference: **ACR** when both ACR and ECR have the same
image (~2.4× faster pull, measured 2026-04-20). Override with `--registry ecr`
or `--registry gitlab` if requested.

If the pull fails with a 401/403 / `denied: requested access to the resource is denied`,
re-auth with the table above.

## Data Source — registry-direct

This skill **does not depend on the speedoflight `commits/index.json`**.
Everything we need comes from the registries themselves + the local git
checkout. Registry is source-of-truth; index is convenience-only.

### Tag format

Both ACR and ECR use the same scheme on `ai-dynamo/dynamo`:

```
<40-char-git-sha>-<suffix>
```

Where `<suffix>` is one of:

| Suffix pattern | Meaning |
|---|---|
| `vllm-dev-cuda12` / `vllm-dev-cuda13` | vllm dev image, CUDA 12/13 variant |
| `vllm-runtime-cuda12` / `vllm-runtime-cuda13` | vllm runtime image |
| `sglang-dev-cuda12` / `sglang-dev-cuda13` | sglang dev/runtime |
| `trtllm-dev-cuda13` | trtllm (CUDA 13 only) |
| `none-dev-cuda13` | bare base image (no framework) |
| `frontend` / `operator` / `snapshot` | service-image singletons (no `<type>-<cuda>` split) |

Regex: `^(?P<sha>[0-9a-f]{40})-(?P<framework>[a-z]+)(-(?P<image_type>dev|runtime)-(?P<cuda>cuda1[23]))?$`

### Listing tags

ACR (preferred — faster pulls, ~2.4× vs ECR for 25 GB images):

```bash
# Quick: tag names only, newest first
az acr repository show-tags --name dynamoci --repository ai-dynamo/dynamo \
    --orderby time_desc --top 200 --output tsv

# With digest + push time (lets us do content-equality matching).
# `acr manifest list-metadata` is the current command; the older
# `acr repository show-manifests` is deprecated but still works.
az acr manifest list-metadata -r dynamoci -n ai-dynamo/dynamo \
    --orderby time_desc --top 200 --output json
# Returns: [{ "digest": "sha256:...", "tags": ["..."], "imageSize": <bytes>,
#             "createdTime": "...", "architecture": "amd64"|"arm64", "os": "linux", ... }, ...]
```

ECR (fallback / authoritative source-of-truth for digests):

```bash
# Tags + digest + push time + size, newest first
aws ecr describe-images --repository-name ai-dynamo/dynamo --region us-west-2 \
    --max-items 200 \
    --query 'reverse(sort_by(imageDetails,&imagePushedAt))[*].{tags:imageTags,digest:imageDigest,pushed:imagePushedAt,bytes:imageSizeInBytes}' \
    --output json
```

GitLab (fallback for arm64 / specific dev variants):

```bash
# Requires docker login first; then list via the GitLab Container Registry API
curl -fsS -H "PRIVATE-TOKEN: $(cat ~/.config/gitlab-token)" \
  "https://gitlab-master.nvidia.com/api/v4/projects/<project_id>/registry/repositories/<repo_id>/tags?per_page=100"
```

### Content-equality across commits (replaces `image_sha_6`)

Two tags whose `digest` field is identical are byte-identical images —
i.e., the `container/` state was the same at both commits. So:

> "different commit, same image" = "different tag, same `imageDigest`".

Use this for fallback matching when the user's HEAD doesn't have its own
image but a later/earlier commit with identical content does.

## Workflow

### Step 0: Ensure auth

```bash
# Refresh the registry token (Claude runs this directly — see Authentication)
az acr login --name dynamoci   # or: aws ecr get-login-password ... | docker login ...
```

If `az acr login` errors with "Please run 'az login'" (or ECR errors with
"Unable to locate credentials"), defer to the user with the one-time setup
hint from the Authentication section. No data fetched yet — fail fast.

### Step 1: Read the user's intent

| User input | framework | image_type | sha_target |
|---|---|---|---|
| "vllm" / "vllm dev" | `vllm` | `dev` | latest |
| "vllm local-dev" | `vllm` | `dev` (will run `local_dev_cmd`) | latest |
| "vllm runtime" | `vllm` | `runtime` | latest |
| "sglang" / "trtllm" / "none" | (as named) | `dev` | latest |
| "for HEAD" / "for my branch" | as above | as above | local `git rev-parse HEAD` |
| "for sha <abc1234>" | as above | as above | given SHA |

Default for ambiguous "latest image": framework=`vllm`, image_type=`dev`,
target=`latest`, registry=`acr`.

### Step 2: List tags + filter to the suffix you want

Build the target suffix from the user's intent:

```python
suffix = f"{framework}-{image_type}-{cuda}"   # e.g. "vllm-dev-cuda13"
# Edge cases: frontend / operator / snapshot have no -<type>-<cuda>; suffix == framework
```

Pull the candidate tag list (newest first) and filter:

```bash
# ACR — prefer this
az acr manifest list-metadata -r dynamoci -n ai-dynamo/dynamo \
    --orderby time_desc --top 200 --output json \
  | jq --arg suf "$suffix" '
      [ .[]
        | { digest, timestamp,
            tags: ([.tags[]? | select(endswith("-"+$suf))]) }
        | select(.tags|length>0)
      ]'
```

(ECR equivalent uses `aws ecr describe-images ... --query` from the table
above; same shape after parsing.)

Each result row has at least one tag of the form `<sha40>-<suffix>`.

### Step 3: Resolve the target SHA against the candidates

If `sha_target == latest` → take the first row (already newest-first).

If `sha_target == <sha>` (from `git rev-parse HEAD` or user-supplied):

1. **Exact-SHA hit** — find a row whose tag starts with the full `<sha>`.
   Done.
2. **Same `digest`** — if (1) misses, look up the digest of any tag at
   `<sha>` (any framework/type — the digest depends on `container/` state,
   not framework wiring; in practice we check the same suffix). If a
   different-SHA row in the candidate list has the same digest, use it.
   Note to user: "different commit, byte-identical image".
3. **Nearest older with image** — walk `git log --format=%H` from the
   target SHA backwards, intersect with the candidate SHA set, take the
   first hit. Report the commit-distance offset.

Always tell the user which case fired. Don't silently substitute.

### Helper: parse a tag

```python
import re
TAG_RE = re.compile(r'^(?P<sha>[0-9a-f]{40})-(?P<rest>.+)$')
def parse_tag(t):
    m = TAG_RE.match(t)
    return m.groupdict() if m else None   # rest is the suffix
```

### Step 4: Construct the pull / local-dev command

Once you have a chosen tag (e.g. `1682167d3c97f0fc8cd405dee51f17288eeaeb55-vllm-dev-cuda13`):

```bash
# Registry path:
#   ACR    → dynamoci.azurecr.io/ai-dynamo/dynamo:<tag>
#   ECR    → 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:<tag>
#   GitLab → gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/dynamo:<tag>

IMAGE="dynamoci.azurecr.io/ai-dynamo/dynamo:<tag>"

# (a) Just pull dev
docker pull "$IMAGE"

# (b) Pull dev + build local-dev (UID-remapped, matches host user)
docker pull "$IMAGE" && \
  curl -sL http://speedoflight.nvidia.com/dynamo/dynamo-utils/container/build_localdev_from_dev.py \
  | python3 - --skip-pull "$IMAGE"
```

Default to (b) — local-dev is what users almost always want for live editing.
Only emit (a) if the user explicitly asked for the dev image without
local-dev wrap.

### Step 5: Picking the registry + surfacing trade-offs

- If both ACR and ECR have the same tag (`digest` equal), prefer ACR
  (~2.4× faster pull). User can override with `--registry ecr|acr|gitlab`.
- If both CUDA 12 and CUDA 13 variants exist for the chosen commit, show
  both with sizes (e.g., "CUDA 12 = 14.3 GB, CUDA 13 = 10.1 GB"). Default
  recommendation: CUDA 13 unless the user is pinned to 12.x.
- If a fallback case fired (same-digest sibling, or nearest-older), say so
  explicitly — no silent substitution.

## Inline Python helper (registry-direct)

```python
import json, re, subprocess

# --- Inputs (set per request) -----------------------------------------------
FRAMEWORK   = "vllm"            # vllm | sglang | trtllm | none | frontend | operator | snapshot
IMAGE_TYPE  = "dev"             # dev | runtime  (ignored for service-images)
CUDA        = "cuda13"          # cuda12 | cuda13  (ignored for service-images)
TARGET_SHA  = "HEAD"            # "latest" | "HEAD" | "<sha>"
REGISTRY    = "acr"             # acr | ecr
REPO_PATH   = "."               # local checkout root for git lookups

# --- Helpers ----------------------------------------------------------------
def sh(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def git_head(repo):
    try: return sh(["git", "-C", repo, "rev-parse", "HEAD"])
    except subprocess.CalledProcessError: return None

def git_log_shas(repo, n=200):
    try: return sh(["git", "-C", repo, "log", f"-n{n}", "--format=%H"]).splitlines()
    except subprocess.CalledProcessError: return []

# --- Build expected suffix --------------------------------------------------
if FRAMEWORK in ("frontend", "operator", "snapshot"):
    SUFFIX = FRAMEWORK
else:
    SUFFIX = f"{FRAMEWORK}-{IMAGE_TYPE}-{CUDA}"

# --- Pull tags from registry ------------------------------------------------
def list_acr():
    # `acr manifest list-metadata` is the current command (preview);
    # the older `acr repository show-manifests` is deprecated.
    raw = sh(["az", "acr", "manifest", "list-metadata",
              "-r", "dynamoci", "-n", "ai-dynamo/dynamo",
              "--orderby", "time_desc", "--top", "200", "--output", "json"])
    return json.loads(raw)
    # [{digest, tags, imageSize, createdTime, architecture, os, ...}, ...]

def list_ecr():
    raw = sh(["aws", "ecr", "describe-images",
              "--repository-name", "ai-dynamo/dynamo",
              "--region", "us-west-2",
              "--max-items", "200",
              "--query", "reverse(sort_by(imageDetails,&imagePushedAt))[*].{digest:imageDigest,tags:imageTags,pushed:imagePushedAt,bytes:imageSizeInBytes}",
              "--output", "json"])
    return json.loads(raw)

manifests = list_acr() if REGISTRY == "acr" else list_ecr()

TAG_RE = re.compile(r"^([0-9a-f]{40})-(.+)$")

# Filter to records whose tag list contains a tag with our suffix
candidates = []
for m in manifests:
    for t in m.get("tags") or []:
        mo = TAG_RE.match(t)
        if not mo: continue
        sha, suf = mo.group(1), mo.group(2)
        if suf == SUFFIX:
            candidates.append({"sha": sha, "tag": t,
                               "digest": m.get("digest"),
                               "size": m.get("imageSize") or m.get("bytes"),
                               "created": m.get("createdTime") or m.get("pushed"),
                               "arch": m.get("architecture")})
            break
# candidates is now ordered newest-first

# --- Resolve target SHA -----------------------------------------------------
target_sha = git_head(REPO_PATH) if TARGET_SHA == "HEAD" else TARGET_SHA
chosen, note = None, None

if target_sha == "latest" or target_sha is None:
    chosen, note = (candidates[0] if candidates else None), "latest"
else:
    # 1) exact match
    chosen = next((c for c in candidates if c["sha"].startswith(target_sha)), None)
    if chosen:
        note = f"exact match for {target_sha[:9]}"
    else:
        # 2) same digest as any tag at target_sha (different commit, same content)
        target_records = [m for m in manifests
                          if any(TAG_RE.match(t) and TAG_RE.match(t).group(1).startswith(target_sha)
                                 for t in (m.get("tags") or []))]
        target_digests = {m["digest"] for m in target_records}
        sibling = next((c for c in candidates if c["digest"] in target_digests), None)
        if sibling:
            chosen = sibling
            note = (f"no exact image for {target_sha[:9]} but tag "
                    f"{sibling['tag']} has identical digest {sibling['digest'][:19]}…")
        else:
            # 3) nearest older commit with image
            sha_set = {c["sha"]: c for c in candidates}
            for offset, sha in enumerate(git_log_shas(REPO_PATH)):
                if sha in sha_set:
                    chosen = sha_set[sha]
                    note = (f"no image at HEAD; nearest older commit with image "
                            f"is {sha[:9]} ({offset} behind)")
                    break

# --- Output -----------------------------------------------------------------
if not chosen:
    print(f"No image found for framework={FRAMEWORK} image_type={IMAGE_TYPE} cuda={CUDA}")
else:
    REGISTRIES = {
        "acr": "dynamoci.azurecr.io/ai-dynamo/dynamo",
        "ecr": "210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo",
    }
    image = f'{REGISTRIES[REGISTRY]}:{chosen["tag"]}'
    print(f"sha:    {chosen['sha'][:9]}")
    print(f"note:   {note}")
    print(f"size:   {chosen.get('size','-')}")
    print(f"image:  {image}")
    print()
    print(f"# Just pull dev:")
    print(f'docker pull "{image}"')
    print()
    print(f"# Pull dev + build local-dev (UID-remapped):")
    print(f'docker pull "{image}" \\')
    print(f'  && curl -sL http://speedoflight.nvidia.com/dynamo/dynamo-utils/container/build_localdev_from_dev.py \\')
    print(f'     | python3 - --skip-pull "{image}"')
```

## Pagination / `--top` sizing

The registries fire roughly one manifest per CI build per arch per CUDA
variant. Recent activity can be dominated by runtime/operator/frontend
builds and push the dev images you care about past the default `--top 200`.

If the suffix-filter returns 0 candidates, retry with `--top 1000` (or
loop with `--orderby time_desc` and increasing `--skip`) before declaring
"image not found".

## Important Notes

- `local_dev_cmd` does pull + local-dev conversion in one shot.
- `pull_cmd` only pulls the dev image (no UID-remap layer).
- `image_sha_6` groups commits with byte-identical container/ state; treat
  as "same image" for matching purposes.
- GitLab images may be arm64-only for some frameworks — check `arch`.
- All hostnames and account IDs are inlined as literals so the user can
  copy-paste any command verbatim — no env file to source.

## Arguments

| Form | Meaning |
|---|---|
| `/dyn-pull-latest-image` | latest vllm dev, default registry ACR |
| `/dyn-pull-latest-image vllm` | latest vllm dev |
| `/dyn-pull-latest-image sglang local-dev` | latest sglang local-dev |
| `/dyn-pull-latest-image vllm runtime` | latest vllm runtime |
| `/dyn-pull-latest-image vllm for HEAD` | image at user's local HEAD (or nearest) |
| `/dyn-pull-latest-image vllm for sha <abc1234>` | image at given SHA (or nearest) |
| `/dyn-pull-latest-image vllm --registry ecr` | force ECR |
