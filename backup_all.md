# backup_all.sh

Cron-driven backup of `keivenc-linux` to `/mnt/sda/keivenc.backup/`.

Single script, three tiers, one destination. Each tier is a superset of the smaller ones — controlled by the flags passed in cron.

## Cron entries

```cron
*/13 * * * *  $UTILS/backup_all.sh >/dev/null 2>&1
0 */6 * * *   $UTILS/backup_all.sh --backup --include-6h >/dev/null 2>&1
0 2 * * *     $UTILS/backup_all.sh --backup --include-6h --include-daily --compress --remove-after-days 45 >/dev/null 2>&1
```

| Frequency | Flags | What it adds vs. the tier below |
|---|---|---|
| every 13 min | (none) | active-work data, corp creds, system config |
| every 6 hours | `--backup --include-6h` | + editor/agent state |
| daily at 02:00 | `--backup --include-6h --include-daily --compress --remove-after-days 45` | + slow/large data + history compression + 45-day retention |

`--backup` is required on the 6 h and daily entries because the script's existing logic disables `DO_BACKUP` whenever `--compress` or `--remove-after-days` is passed without an explicit `--backup`.

## Tier 1 — every 13 min (~2.8 GB scan, ~tens of MB delta)

Backs up `TARGETS` + `SINGLE_FILES` + `run_system_backup()`.

| Source | Size | Dest under `/mnt/sda/keivenc.backup/` |
|---|---|---|
| `~/dynamo` (with `.rsyncrules` excludes) | ~2.6 G after excludes (590 G raw) | `dynamo/` |
| `~/.config` | 155 M | `.config/` |
| `~/.ssh` | 56 K | `.ssh/` |
| `~/.ngc` | <1 M | `.ngc/` |
| `~/.aws` | 60 K | `.aws/` |
| `~/.azure` | 2.3 M | `.azure/` |
| `~/.kube` | 940 K | `.kube/` |
| `~/.cisco` | 72 K | `.cisco/` |
| `~/ai-config` | 2.6 M | `ai-config/` |
| `~/.gitconfig`, `.bashrc`, `.bash_profile`, `.profile`, `.zshrc`, `.claude.json` | ~50 K | `dotfiles/` |
| `/etc/{ufw,nginx,default/grub,default/grub.d,systemd/system,fstab,crypttab,netplan,sudoers.d,hosts,hostname}` + `/var/spool/cron/crontabs/keivenc` | ~600 K | `system/` (sudo'd; preserves absolute paths) |

The system backup needs `NOPASSWD` sudo (already configured); without it the function logs a warning and no-ops.

## Tier 2 — every 6 hours (adds ~775 MB)

`TARGETS_6H` — editor/agent state and corp credentials that change a few times a day, not every minute.

| Source | Size | Dest |
|---|---|---|
| `~/.claude` | 393 M | `.claude/` |
| `~/.cursor` | 22 M | `.cursor/` |
| `~/.tsh` | 239 M | `.tsh/` (Teleport SSH cert cache) |
| `~/.slack` | 18 M | `.slack/` |
| `~/nvsec-env` | 46 M | `nvsec-env/` |

Cumulative every-6 h scan: ~3.6 GB.

## Tier 3 — daily at 02:00 (adds ~4.0 GB)

`TARGETS_DAILY` — large or slow-changing data where 24 h resolution is fine.

| Source | Size | Dest |
|---|---|---|
| `~/.gnupg` | 104 K | `.gnupg/` |
| `~/.docker` | 36 M | `.docker/` |
| `~/.cache/dynamo-utils` | ~2.4 G after `*.tmp` exclude (19 G raw) | `.cache/dynamo-utils/` |
| `~/.local/share` | 1.6 G | `.local/share/` |

Cumulative daily scan: ~7.6 GB.

Plus maintenance after the rsync:

- `--compress` — gzips any prior calendar day's `backup_history/<YYYYMMDD_HHMMSS>/` snapshots into a single `<YYYYMMDD>.tgz`. Typical 3–5× shrink.
- `--remove-after-days 45` — deletes `backup_history/` dirs and `.tgz` archives older than 45 days.

## Versioning

Every backup run also writes a timestamped `backup_history/<YYYYMMDD_HHMMSS>/` snapshot containing **only files that were changed or deleted in that run** (rsync's `--backup --backup-dir`). So if a file was overwritten or removed at 14:39 today, the previous version sits in `backup_history/20260506_143901/<dest_path>/...`. That's the rolling 45-day window.

Steady-state size of `backup_history/`: ~30–40 GB on a 3.6 TB volume with ~1.7 TB free.

## Skipped (regenerable, by design)

| Path | Size | Why skip |
|---|---|---|
| `~/.cache/huggingface` | 520 G | Symlinked to `/mnt/sda` already; explicit exclude in script |
| `~/bin/Linux.x86_64/venv.3.12` + `~/venv` | ~16 G | `pip install` rebuilds |
| `~/.rustup` + `~/.cargo` | 8.6 G | `rustup install` / `cargo build` rebuilds |
| `~/.cursor-server/{bin,extensions}` | 2 G | Cursor re-downloads on next remote connect |
| `~/.cache/{vllm,pip,uv,pyright-python,pre-commit,gh,flashinfer}` | ~20 G | Caches rebuild themselves |
| `~/.nvm`, `~/.npm`, `~/.krew`, `~/snap` | ~400 M | Package managers rebuild |

Total skipped: ~570 GB.

## Recovery scenarios

| Failure | Lost | Survived | Recovery |
|---|---|---|---|
| nvme0n1 (system disk) fails | `/`, `/home`, `/etc`, `/boot` | All backups (on `/mnt/sda`) + Docker root + `/mnt/sda/keivenc/` workspace | Reinstall Ubuntu, set up LUKS, run cinc-client to push corp config back, then rsync `/mnt/sda/keivenc.backup/system/etc/*` and `/home/keivenc/*` from backup. The `system/` backup has the firewall, fstab, grub params, custom systemd units, and crontab needed to recover the stability tunings and automation. |
| `/mnt/sda` fails | All backup history (33 G), Docker root (all images/containers/volumes incl. Prometheus history), `/mnt/sda/keivenc/` workspace | Live `/home`, `/etc`, OS | Replace disk; recreate `/mnt/sda` mount; redo Docker state; restore `dynamo_ci`/`gputest` from git. Lose all Prometheus/timeseries history. |
| Both disks fail / fire / theft | Everything | Nothing on this box | Off-machine restore: GitHub remotes for code; corp IT for config; **no off-machine backup currently exists** — this is the largest open gap. |

## Operational notes

- `--dry-run` exits 1 after the first target due to a pre-existing pipefail interaction with `find` on a non-created `target_history_dir`. Real (non-dry) runs work fine. Don't rely on `--dry-run` for verification.
- The script uses `set -euo pipefail` and `umask 077`. Backup destination dirs end up `chmod 700`; single files `chmod 600`.
- Backup logs: `~/dynamo/logs/<YYYY-MM-DD>/backup.log` (and per-target sublogs in the same dir).
- Per-run snapshot inspection: `ls -la /mnt/sda/keivenc.backup/backup_history/ | tail`.
