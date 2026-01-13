# Remote Branches Crontab

This document explains the cron schedule for `show_remote_branches.py` updates.

## Schedule

### Working Hours (8am-6pm PT)
- **Frequency**: Every 1 minute
- **Users**: kthui, keivenchang
- **Time (PT)**: 8:00am - 5:59pm
- **Time (UTC)**: 4:00pm - 1:59am (next day)

### Off Hours (6pm-8am PT)
- **Frequency**: Every 20 minutes
- **Users**: kthui, keivenchang  
- **Time (PT)**: 6:00pm - 7:59am
- **Time (UTC)**: 2:00am - 3:59pm

## Crontab Entries

```cron
# Working hours (8am-6pm PT / 4pm-2am UTC): every 1 minute
* 16-23 * * * NVIDIA_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-remote-branches
* 0-1 * * * NVIDIA_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-remote-branches

# Off hours (6pm-8am PT / 2am-4pm UTC): every 20 minutes
*/20 2-15 * * * NVIDIA_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils/cron_log.sh remote_prs_offhours $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-remote-branches
```

## Output Locations

- **kthui**: `$HOME/dynamo/speedoflight/users/kthui/index.html`
- **keivenchang**: `$HOME/dynamo/speedoflight/users/keivenchang/index.html`

## Logs

- Working hours: `$HOME/dynamo/logs/YYYY-MM-DD/remote_prs_working.log`
- Off hours: `$HOME/dynamo/logs/YYYY-MM-DD/remote_prs_offhours.log`
- Script logs: `$HOME/dynamo/logs/YYYY-MM-DD/show_remote_branches.log`

## Notes

1. Times are in **UTC** (server time)
2. Pacific Time (PT) includes both PST and PDT (timezone-aware)
3. Working hours span two UTC days:
   - Same day: 16:00-23:59 UTC
   - Next day: 00:00-01:59 UTC
4. GitHub API calls are limited by `MAX_GITHUB_API_CALLS` env var (if set)
5. Uses same caching as local branches (`~/.cache/dynamo-utils/`)
6. **Remote branches now have IDENTICAL structure to local branches** (as of 2026-01-08):
   - Full CI job hierarchy with parent-child relationships
   - Collapsible UserNode for each GitHub user
   - Same PASSED/FAILED status pills and job details
   - Reuses exact same rendering logic from `show_local_branches.py`

## Installation

Add these lines to your crontab:

```bash
crontab -e
# Paste the cron entries from above
```

Or use the provided file:

```bash
cat dynamo-utils/html_pages/crontab_remote_branches.txt >> /tmp/mycron
crontab /tmp/mycron
```


