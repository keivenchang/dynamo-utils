#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Official backup entry point for this machine.
# Replaces the old pattern of per-target backup cron entries.
#
# Default mode:
#   - back up all configured targets below
#   - update versioned backup history under /mnt/sda/keivenc.backup/backup_history/
#
# Operations-only mode:
#   - if maintenance flags are provided with no explicit --backup, skip rsync
#   - perform only compression / uncompression / retention cleanup
#
# Usage (cron — normal backups):
#   */6 * * * * $DYNAMO_HOME/dynamo-utils.dev/backup_all.sh >/dev/null 2>&1
#
# Usage (cron — maintenance):
#   0 2 * * * $DYNAMO_HOME/dynamo-utils.dev/backup_all.sh --compress --remove-after-days 45 >/dev/null 2>&1

set -euo pipefail

# Backups must not be world-readable (credentials, keys, configs)
umask 077

BACKUP_ROOT="/mnt/sda/keivenc.backup"
BACKUP_HISTORY_ROOT="$BACKUP_ROOT/backup_history"
DYNAMO_HOME="${DYNAMO_HOME:-$HOME/dynamo}"
LOGS_DIR="${LOGS_DIR:-$DYNAMO_HOME/logs}"
TODAY="$(date +%Y-%m-%d)"
DAY_LOG_DIR="$LOGS_DIR/$TODAY"
LOG_FILE="${LOG_FILE:-$DAY_LOG_DIR/backup.log}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

mkdir -p "$DAY_LOG_DIR"

DO_BACKUP=true
BACKUP_EXPLICIT=false
COMPRESS=false
UNCOMPRESS=false
REMOVE_AFTER_DAYS=""
DRY_RUN=false
FAILED_TARGETS=()

# Format: source_path|dest_relative_path
TARGETS=(
    "${DYNAMO_HOME}|nvidia"
    "$HOME/.config|.config"
    "$HOME/.ssh|.ssh"
    "$HOME/.claude|.claude"
    "$HOME/.cursor|.cursor"
    "$HOME/.ngc|.ngc"
    # NOT ~/.cache/huggingface — it is already symlinked to /mnt/sda
)

# Single files copied separately after directory backups.
SINGLE_FILES=(
    "$HOME/.gitconfig"
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
    "$HOME/.profile"
    "$HOME/.zshrc"
)

show_usage() {
    cat <<'EOF'
Usage: backup_all.sh [OPTIONS]

Default behavior with no options:
  Back up all configured targets.

If maintenance flags are provided and --backup is NOT specified:
  Run in operations-only mode (skip rsync backup).

OPTIONS:
  --backup                    Force backup mode, even with maintenance flags
  --compress                  Compress old backup history directories by date
  --uncompress                Uncompress all backup history archives
  --remove-after-days <N>     Remove backup history older than N days
  --dry-run, --dryrun         Show what would be done without making changes
  -h, --help                  Show this help message

EXAMPLES:
  backup_all.sh
  backup_all.sh --compress --remove-after-days 45
  backup_all.sh --backup --compress --remove-after-days 45
  backup_all.sh --uncompress --dry-run
EOF
}

cleanup_working_files() {
    if [ -d "$BACKUP_HISTORY_ROOT" ]; then
        rm -f "$BACKUP_HISTORY_ROOT"/*.tgz.working 2>/dev/null || true
    fi
}

trap cleanup_working_files EXIT INT TERM

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backup)
            DO_BACKUP=true
            BACKUP_EXPLICIT=true
            shift
            ;;
        --compress)
            COMPRESS=true
            shift
            ;;
        --uncompress)
            UNCOMPRESS=true
            shift
            ;;
        --remove-after-days)
            REMOVE_AFTER_DAYS="${2:-}"
            shift 2
            ;;
        --dry-run|--dryrun)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done

if [[ "$COMPRESS" == true || "$UNCOMPRESS" == true || -n "$REMOVE_AFTER_DAYS" ]] && [[ "$BACKUP_EXPLICIT" == false ]]; then
    DO_BACKUP=false
fi

if [[ "$COMPRESS" == true && "$UNCOMPRESS" == true ]]; then
    echo "Error: --compress and --uncompress cannot be used together" >&2
    exit 1
fi

if [[ -n "$REMOVE_AFTER_DAYS" ]]; then
    if ! [[ "$REMOVE_AFTER_DAYS" =~ ^[0-9]+$ ]] || [[ "$REMOVE_AFTER_DAYS" -le 0 ]]; then
        echo "Error: --remove-after-days must be a positive integer" >&2
        exit 1
    fi
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "*** DRY RUN MODE - No changes will be made ***"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - DRY RUN MODE" >> "$LOG_FILE"
fi

build_exclude_opts() {
    local source_dir="$1"
    local -n out_opts="$2"
    local rsyncrules_file="$source_dir/.rsyncrules"

    out_opts=()
    if [[ -f "$rsyncrules_file" ]]; then
        echo "Reading exclusions from: $rsyncrules_file"
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
                continue
            fi
            if [[ "$line" =~ ^\+ ]]; then
                local pattern="${line#\+}"
                pattern="${pattern#"${pattern%%[![:space:]]*}"}"
                out_opts+=("--include=$pattern")
            elif [[ "$line" =~ ^\- ]]; then
                local pattern="${line#-}"
                pattern="${pattern#"${pattern%%[![:space:]]*}"}"
                out_opts+=("--exclude=$pattern")
            else
                out_opts+=("--exclude=$line")
            fi
        done < "$rsyncrules_file"
    else
        echo "Warning: .rsyncrules file not found at $rsyncrules_file"
        echo "Using default exclusions"
        out_opts=(
            "--exclude=*.pyc"
            "--exclude=__pycache__/"
            "--exclude=.pytest_cache/"
            "--exclude=.mypy_cache/"
            "--exclude=.ruff_cache/"
            "--exclude=*.egg-info/"
            "--exclude=.venv/"
            "--exclude=venv/"
            "--exclude=node_modules/"
            "--exclude=target/"
            "--exclude=build/"
            "--exclude=dist/"
            "--exclude=*.o"
            "--exclude=*.so"
            "--exclude=*.dylib"
            "--exclude=.DS_Store"
            "--exclude=.git/objects/"
            "--exclude=.git/lfs/"
            "--exclude=.git/FETCH_HEAD"
            "--exclude=.git/HEAD"
            "--exclude=.git/index"
            "--exclude=.git/logs/"
            "--exclude=*.log"
            "--exclude=/tmp/"
            "--exclude=*.tmp"
            "--exclude=*.swp"
            "--exclude=*.swo"
            "--exclude=*~"
            "--exclude=*.bak"
            "--exclude=core"
            "--exclude=core.*"
            "--exclude=.cache/huggingface/"
        )
    fi
}

remove_empty_history_dirs() {
    local path="$1"
    local current="$path"
    while [[ "$current" != "$BACKUP_HISTORY_ROOT" && "$current" != "/" ]]; do
        rmdir "$current" 2>/dev/null || break
        current="$(dirname "$current")"
    done
}

run_backup_target() {
    local src="$1"
    local dest_rel="$2"
    local dest_dir="$BACKUP_ROOT/$dest_rel"
    local target_history_dir="$BACKUP_HISTORY_ROOT/$TIMESTAMP/$dest_rel"
    local rc=0

    if [[ ! -d "$src" ]]; then
        echo "Skipping missing directory: $src"
        return 0
    fi

    echo "Starting backup: $src -> $dest_dir"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting backup: $src -> $dest_dir" >> "$LOG_FILE"
    echo "Backup history: $target_history_dir"

    local -a exclude_opts=()
    build_exclude_opts "$src" exclude_opts

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would create directories: $dest_dir and $target_history_dir"
    else
        mkdir -p "$dest_dir" "$target_history_dir"
    fi

    local -a rsync_cmd=(
        rsync -av --delete --backup "--backup-dir=$target_history_dir"
    )
    if [[ "$DRY_RUN" == true ]]; then
        rsync_cmd+=(--dry-run)
    fi
    rsync_cmd+=("${exclude_opts[@]}" "$src/" "$dest_dir/")

    printf '+ '
    printf '%q ' "${rsync_cmd[@]}"
    printf '\n'

    set +e
    "${rsync_cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -eq 0 || "$rc" -eq 23 ]]; then
        if [[ "$rc" -eq 23 ]]; then
            echo "Backup completed with warnings (some files/attrs were not transferred)"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup completed with warnings for $dest_rel (exit code 23)" >> "$LOG_FILE"
        else
            echo "Backup completed successfully!"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup completed successfully for $dest_rel" >> "$LOG_FILE"
        fi

        local history_count
        history_count=$(find "$target_history_dir" -type f 2>/dev/null | wc -l)
        if [[ "$history_count" -gt 0 ]]; then
            local history_size
            history_size=$(du -sh "$target_history_dir" 2>/dev/null | awk '{print $1}')
            echo "Changed/deleted files backed up: $history_count files ($history_size)"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup history for $dest_rel: $history_count files in $target_history_dir" >> "$LOG_FILE"
        else
            echo "No files changed or deleted - backup history empty"
            if [[ "$DRY_RUN" == true ]]; then
                echo "[DRY RUN] Would remove empty history directory: $target_history_dir"
            else
                remove_empty_history_dirs "$target_history_dir"
            fi
        fi

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY RUN] Would chmod 700 $dest_dir"
        else
            chmod 700 "$dest_dir"
        fi
    else
        echo "Backup failed with exit code $rc for $src"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup failed for $dest_rel with exit code $rc" >> "$LOG_FILE"
        FAILED_TARGETS+=("$dest_rel")
    fi
}

copy_single_files() {
    local dotfiles_dir="$BACKUP_ROOT/dotfiles"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would create dotfiles directory: $dotfiles_dir"
    else
        mkdir -p "$dotfiles_dir"
        chmod 700 "$dotfiles_dir"
    fi

    for f in "${SINGLE_FILES[@]}"; do
        if [[ -f "$f" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "[DRY RUN] Would copy $f -> $dotfiles_dir/"
            else
                \cp -f "$f" "$dotfiles_dir/"
                chmod 600 "$dotfiles_dir/$(basename "$f")"
            fi
        fi
    done
}

compress_history() {
    echo "Compressing old backup history directories by date..."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting compression of old backups" >> "$LOG_FILE"

    local today_date
    today_date="$(date '+%Y%m%d')"

    [[ -d "$BACKUP_HISTORY_ROOT" ]] || return 0

    rm -f "$BACKUP_HISTORY_ROOT"/*.tgz.working 2>/dev/null || true

    local dates_to_compress
    dates_to_compress=$(find "$BACKUP_HISTORY_ROOT" -maxdepth 1 -type d -name '[0-9]*_[0-9]*' | \
        sed 's|.*/||' | \
        cut -d'_' -f1 | \
        sort -u | \
        while read -r date; do
            if [[ "$date" -lt "$today_date" ]]; then
                echo "$date"
            fi
        done)

    for backup_date in $dates_to_compress; do
        local archive_path="$BACKUP_HISTORY_ROOT/${backup_date}.tgz"
        local working_archive="${archive_path}.working"
        local date_dirs

        if [[ -f "$archive_path" ]]; then
            echo "Already compressed: ${backup_date}.tgz"
            continue
        fi

        date_dirs=$(find "$BACKUP_HISTORY_ROOT" -maxdepth 1 -type d -name "${backup_date}_*" | sed 's|.*/||' | sort)
        [[ -n "$date_dirs" ]] || continue

        local dir_count
        dir_count=$(echo "$date_dirs" | wc -l)

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY RUN] Would compress $dir_count directories for date $backup_date -> ${backup_date}.tgz"
            echo "$date_dirs" | while read -r dirname; do
                echo "  - $dirname"
            done
            continue
        fi

        echo "Compressing $dir_count directories for date $backup_date:"
        echo "$date_dirs" | while read -r dirname; do
            echo "  - $dirname"
        done

        set -x
        tar -I pigz -cf "$working_archive" -C "$BACKUP_HISTORY_ROOT" $date_dirs
        local tar_exit=$?
        { set +x; } 2>/dev/null

        if [[ "$tar_exit" -eq 0 ]]; then
            set -x
            \mv -f "$working_archive" "$archive_path"
            { set +x; } 2>/dev/null

            while read -r dirname; do
                [[ -n "$dirname" ]] || continue
                set -x
                rm -rf "$BACKUP_HISTORY_ROOT/$dirname"
                { set +x; } 2>/dev/null
            done <<< "$date_dirs"

            local compressed_size
            compressed_size=$(du -sh "$archive_path" 2>/dev/null | awk '{print $1}')
            echo "Compressed to: ${backup_date}.tgz ($compressed_size)"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Compressed $dir_count directories for $backup_date to ${backup_date}.tgz" >> "$LOG_FILE"
        else
            echo "Failed to compress directories for date: $backup_date"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed to compress $backup_date" >> "$LOG_FILE"
            rm -f "$working_archive"
        fi
    done
}

uncompress_history() {
    echo "Uncompressing backup history archives..."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting uncompression of backup archives" >> "$LOG_FILE"

    [[ -d "$BACKUP_HISTORY_ROOT" ]] || return 0

    for archive in "$BACKUP_HISTORY_ROOT"/[0-9]*.tgz; do
        [[ -f "$archive" ]] || continue

        local archive_filename archive_size tar_exit extracted_count
        archive_filename=$(basename "$archive" .tgz)
        archive_size=$(du -sh "$archive" 2>/dev/null | awk '{print $1}')

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY RUN] Would uncompress: $archive ($archive_size)"
            continue
        fi

        echo "Uncompressing: $archive ($archive_size)"
        set -x
        tar -I pigz -xf "$archive" -C "$BACKUP_HISTORY_ROOT"
        tar_exit=$?
        { set +x; } 2>/dev/null

        if [[ "$tar_exit" -eq 0 ]]; then
            set -x
            rm -f "$archive"
            { set +x; } 2>/dev/null

            extracted_count=$(find "$BACKUP_HISTORY_ROOT" -maxdepth 1 -type d -name "${archive_filename}_*" | wc -l)
            echo "Uncompressed $extracted_count directories from: ${archive_filename}.tgz"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Uncompressed ${archive_filename}.tgz" >> "$LOG_FILE"
        else
            echo "Failed to uncompress: $archive"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed to uncompress ${archive_filename}.tgz" >> "$LOG_FILE"
        fi
    done
}

remove_old_history() {
    [[ -n "$REMOVE_AFTER_DAYS" ]] || return 0

    echo "Removing backup history older than $REMOVE_AFTER_DAYS days..."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Removing backups older than $REMOVE_AFTER_DAYS days" >> "$LOG_FILE"

    [[ -d "$BACKUP_HISTORY_ROOT" ]] || return 0

    local cutoff_date
    cutoff_date=$(date -d "$REMOVE_AFTER_DAYS days ago" '+%Y%m%d' 2>/dev/null || date -v-"${REMOVE_AFTER_DAYS}d" '+%Y%m%d' 2>/dev/null)

    for backup_dir in "$BACKUP_HISTORY_ROOT"/*/; do
        [[ -d "$backup_dir" ]] || continue
        local backup_dirname backup_date
        backup_dirname=$(basename "$backup_dir")
        backup_date=$(echo "$backup_dirname" | cut -d'_' -f1)
        if [[ "$backup_date" -lt "$cutoff_date" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "[DRY RUN] Would remove directory: $backup_dir"
            else
                echo "Removing old directory: $backup_dir"
                rm -rf "$backup_dir"
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Removed old directory $backup_dirname" >> "$LOG_FILE"
            fi
        fi
    done

    for backup_archive in "$BACKUP_HISTORY_ROOT"/*.tgz; do
        [[ -f "$backup_archive" ]] || continue
        local backup_filename backup_date archive_size
        backup_filename=$(basename "$backup_archive" .tgz)
        backup_date=$(echo "$backup_filename" | cut -d'_' -f1)
        if [[ "$backup_date" -lt "$cutoff_date" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "[DRY RUN] Would remove archive: $backup_archive"
            else
                archive_size=$(du -sh "$backup_archive" 2>/dev/null | awk '{print $1}')
                echo "Removing old archive: $backup_archive ($archive_size)"
                rm -f "$backup_archive"
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Removed old archive ${backup_filename}.tgz" >> "$LOG_FILE"
            fi
        fi
    done
}

if [[ "$DO_BACKUP" == true ]]; then
    mkdir -p "$HOME/.config"
    crontab -l > "$HOME/.config/crontab.backup" 2>/dev/null || true

    for entry in "${TARGETS[@]}"; do
        run_backup_target "${entry%%|*}" "${entry##*|}"
    done

    copy_single_files
fi

if [[ "$COMPRESS" == true ]]; then
    compress_history
fi

if [[ "$UNCOMPRESS" == true ]]; then
    uncompress_history
fi

remove_old_history

if [[ "$DO_BACKUP" == true && "${#FAILED_TARGETS[@]}" -gt 0 ]]; then
    echo "Backup completed with failures for: ${FAILED_TARGETS[*]}" >&2
    exit 1
fi

echo "Backup history location: $BACKUP_HISTORY_ROOT"
echo "Log file: $LOG_FILE"
