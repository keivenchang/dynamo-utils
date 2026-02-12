#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backfill JSON build reports from existing HTML report files.

For each YYYY-MM-DD.<sha9>.report.html found under the logs directory,
creates a corresponding YYYY-MM-DD.<sha9>.json by parsing the HTML.

One-time migration script. After this, build_images.py writes JSON natively.

Usage:
    python3 scripts/backfill_build_reports.py ~/dynamo/dynamo_ci/logs [--dry-run] [--force]
"""
import argparse
import glob
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common_build_report import (
    BuildReport,
    CommitInfo,
    FrameworkResult,
    RegistryImage,
    TargetResult,
    TaskResult,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)
FRAMEWORKS = ["none", "sglang", "trtllm", "vllm"]


def _parse_size(num_str: str, unit: str) -> Optional[int]:
    mult = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    try:
        return int(float(num_str) * mult.get(unit, 1))
    except ValueError:
        return None


def parse_html_report(html_path: Path) -> Optional[BuildReport]:
    """Parse HTML report and return a BuildReport."""
    content = html_path.read_text(errors="replace")
    fname = html_path.name
    m = re.match(r"(\d{4}-\d{2}-\d{2})\.([a-f0-9]{7,9})\.report\.html$", fname)
    if not m:
        logger.warning("Filename mismatch: %s", fname)
        return None
    date_str, sha_short = m.group(1), m.group(2)

    sha_full_m = re.search(r'github\.com/ai-dynamo/dynamo/commit/([a-f0-9]{7,40})', content)
    sha_full = sha_full_m.group(1) if sha_full_m else sha_short

    overall_status = "PASS"
    if "TESTS FAILED" in content:
        overall_status = "FAIL"
    elif "BUILD INTERRUPTED" in content:
        overall_status = "KILLED"
    elif "BUILD IN PROGRESS" in content:
        overall_status = "RUNNING"

    report_gen_m = re.search(r'Report Generated:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
    report_generated = report_gen_m.group(1) if report_gen_m else ""

    def extract_count(label: str) -> int:
        pat = re.compile(r'<div class="number">(\d+)</div>\s*<div class="label">' + label, re.I)
        m2 = pat.search(content)
        return int(m2.group(1)) if m2 else 0

    total_tasks = extract_count("Total Tasks")
    succeeded = extract_count("Succeeded")
    failed_count = extract_count("Failed")
    skipped_count = extract_count("Skipped")
    killed_count = extract_count("Killed")

    commit_info = CommitInfo(sha_short=sha_short, sha_full=sha_full)
    author_m = re.search(r'<strong>Author:</strong>\s*(.+?)(?:</p>|<br)', content)
    if author_m:
        commit_info.author = author_m.group(1).strip()
    date_m = re.search(r'<strong>Date:</strong>\s*(.+?)(?:</p>|<br)', content)
    if date_m:
        commit_info.date = date_m.group(1).strip()
    msg_m = re.search(r'<strong>Message:</strong>\s*(.+?)(?:</p>|<br)', content)
    if msg_m:
        commit_info.message = msg_m.group(1).strip()
    pr_m = re.search(r'<strong>PR:</strong>\s*<a[^>]*>#?(\d+)</a>', content)
    if pr_m:
        commit_info.pr_number = int(pr_m.group(1))
    changes_m = re.search(r'\+(\d+)/-(\d+)\s+lines', content)
    if changes_m:
        commit_info.insertions = int(changes_m.group(1))
        commit_info.deletions = int(changes_m.group(2))

    framework_sections = re.split(r'<div class="framework-header"[^>]*>', content)
    framework_results: List[FrameworkResult] = []
    parsed_frameworks: set = set()

    for section in framework_sections[1:]:
        fw_name_m = re.match(r'\s*([A-Za-z_]+)\s*</div>', section)
        if not fw_name_m:
            continue
        framework = fw_name_m.group(1).strip().lower()
        if framework not in FRAMEWORKS or framework in parsed_frameworks:
            continue
        parsed_frameworks.add(framework)

        targets: List[TargetResult] = []
        registry_images: List[RegistryImage] = []
        next_fw = section.find('<div class="framework-header"')
        sec = section[:next_fw] if next_fw > 0 else section
        row_splits = re.split(r'<div class="chart-cell chart-target">([^<]+)</div>', sec)

        i = 1
        while i < len(row_splits) - 1:
            target_name = row_splits[i].strip()
            row_content = row_splits[i + 1]
            i += 2

            if target_name == "dev-upload":
                loc_m = re.search(r"(gitlab-master\.nvidia\.com:5005/[^<\s'\"()]+)", row_content)
                if loc_m:
                    location = loc_m.group(1).strip()
                    image_name = location.rsplit("/", 1)[-1] if "/" in location else location
                    tag = image_name.split(":", 1)[-1] if ":" in image_name else image_name
                    push_status = "SKIP"
                    if "UPLOADED" in row_content or "PASS" in row_content:
                        push_status = "PASS"
                    elif "FAIL" in row_content:
                        push_status = "FAIL"
                    registry_images.append(RegistryImage(
                        location=location, image_name=image_name, tag=tag,
                        framework=framework, target="dev", push_status=push_status,
                    ))
                continue

            # Normal target
            img_names = re.findall(r'font-family: monospace[^>]*>([\w:.\-]+)', row_content)
            input_image = img_names[0] if len(img_names) >= 1 and img_names[0] != "-" else None
            output_image = img_names[1] if len(img_names) >= 2 else None

            def parse_task(chunk: str) -> Optional[TaskResult]:
                status, duration, prev = None, None, None
                if "PASS" in chunk:
                    if "prev:" in chunk.lower():
                        status, prev = "SKIP", "PASS"
                    else:
                        status = "PASS"
                elif "FAIL" in chunk:
                    if "prev:" in chunk.lower():
                        status, prev = "SKIP", "FAIL"
                    else:
                        status = "FAIL"
                elif "KILLED" in chunk:
                    status = "KILLED"
                dur_m2 = re.search(r'\((\d+\.?\d*)s\)', chunk)
                if dur_m2:
                    duration = float(dur_m2.group(1))
                if status is None and chunk.strip() in ("-", ""):
                    return None
                return TaskResult(status=status or "SKIP", duration_s=duration, prev_status=prev) if status else None

            cells = re.findall(r'<div class="chart-cell chart-status[^"]*"[^>]*>(.*?)</div>', row_content, re.DOTALL)
            build_r = parse_task(cells[0]) if len(cells) > 0 else None
            comp_r = parse_task(cells[1]) if len(cells) > 1 else None
            san_r = parse_task(cells[2]) if len(cells) > 2 else None

            size_m2 = re.search(r'chart-timing">(\d+\.?\d*)\s*(GB|MB|KB|TB)\s*</div>', row_content)
            img_size = _parse_size(size_m2.group(1), size_m2.group(2)) if size_m2 else None

            targets.append(TargetResult(
                target=target_name, build=build_r, compilation=comp_r, sanity=san_r,
                image_name=output_image, image_size_bytes=img_size, input_image=input_image,
            ))

        # Populate registry image size_bytes from the dev target (parsed after dev-upload row)
        dev_size = None
        for t in targets:
            if t.target == "dev" and t.image_size_bytes:
                dev_size = t.image_size_bytes
                break
        if dev_size:
            for img in registry_images:
                if img.size_bytes is None:
                    img.size_bytes = dev_size

        framework_results.append(FrameworkResult(framework=framework, targets=targets, registry_images=registry_images))

    for fw in FRAMEWORKS:
        if fw not in parsed_frameworks:
            framework_results.append(FrameworkResult(framework=fw))

    fw_order = {fw: i for i, fw in enumerate(FRAMEWORKS)}
    framework_results.sort(key=lambda fr: fw_order.get(fr.framework, 999))

    return BuildReport(
        schema_version=SCHEMA_VERSION, sha_short=sha_short, sha_full=sha_full,
        build_date=date_str, report_generated=report_generated, overall_status=overall_status,
        total_tasks=total_tasks, succeeded=succeeded, failed=failed_count,
        skipped=skipped_count, killed=killed_count, commit=commit_info,
        frameworks=framework_results,
    )


def backfill_directory(logs_dir: Path, dry_run: bool = False, force: bool = False) -> Tuple[int, int, int]:
    processed, skipped, errors = 0, 0, 0
    html_files = sorted(glob.glob(str(logs_dir / "*" / "*.report.html")))
    logger.info("Found %d HTML report files in %s", len(html_files), logs_dir)

    for html_path_str in html_files:
        html_path = Path(html_path_str)
        json_path = html_path.parent / html_path.name.replace(".report.html", ".json")
        if json_path.exists() and not force:
            skipped += 1
            continue
        try:
            report = parse_html_report(html_path)
            if report is None:
                errors += 1
                continue
            if dry_run:
                n_imgs = len(report.all_registry_images())
                logger.info("[DRY-RUN] %s -> %d frameworks, %d registry images", json_path.name, len(report.frameworks), n_imgs)
            else:
                report.to_file(json_path)
                logger.info("Written: %s", json_path.name)
            processed += 1
        except Exception as e:
            logger.error("Error processing %s: %s", html_path, e)
            errors += 1

    return processed, skipped, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill JSON build reports from HTML")
    parser.add_argument("logs_dir", type=Path, help="Root logs directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    if not args.logs_dir.is_dir():
        logger.error("Not a directory: %s", args.logs_dir)
        return 1
    p, s, e = backfill_directory(args.logs_dir, args.dry_run, args.force)
    logger.info("Done: %d processed, %d skipped, %d errors", p, s, e)
    return 0 if e == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
