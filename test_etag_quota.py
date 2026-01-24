#!/usr/bin/env python3
"""Test if ETag 304 responses consume GitHub API quota"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from common_github import GitHubAPIClient

def main():
    client = GitHubAPIClient()
    
    # Get quota BEFORE
    before = client.get_core_rate_limit_info()
    print(f"BEFORE: remaining={before['remaining']}, used={before['limit'] - before['remaining']}")
    
    # Make an initial call to get an ETag
    print("\n1. Making initial call to get ETag...")
    url = "https://api.github.com/repos/ai-dynamo/dynamo/commits/main"
    resp = client._rest_get(url, timeout=10)
    etag = resp.headers.get('ETag')
    print(f"   Status: {resp.status_code}, ETag: {etag}")
    
    # Get quota AFTER first call
    after_first = client.get_core_rate_limit_info()
    print(f"   Quota after 1st call: remaining={after_first['remaining']}")
    print(f"   Consumed: {before['remaining'] - after_first['remaining']} calls")
    
    # Now make 10 calls with the ETag (should all return 304)
    print(f"\n2. Making 10 ETag requests (expecting 304 Not Modified)...")
    etag_statuses = []
    for i in range(10):
        resp = client._rest_get(url, timeout=10, etag=etag)
        etag_statuses.append(resp.status_code)
        print(f"   Call {i+1}: status={resp.status_code}")
    
    # Get quota AFTER ETag calls
    after_etags = client.get_core_rate_limit_info()
    print(f"\nAFTER 10 ETag calls: remaining={after_etags['remaining']}")
    print(f"Consumed by 10 ETag calls: {after_first['remaining'] - after_etags['remaining']} calls")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Initial call (200 OK): consumed {before['remaining'] - after_first['remaining']} quota")
    print(f"  10 ETag calls ({etag_statuses[0]} status): consumed {after_first['remaining'] - after_etags['remaining']} quota")
    print(f"  Total consumed: {before['remaining'] - after_etags['remaining']} calls")
    print(f"{'='*60}")
    
    if all(s == 304 for s in etag_statuses):
        if after_first['remaining'] == after_etags['remaining']:
            print("\n✅ CONFIRMED: ETag 304 responses do NOT consume API quota!")
        else:
            print(f"\n❌ UNEXPECTED: ETag 304 responses consumed {after_first['remaining'] - after_etags['remaining']} quota!")
    else:
        print(f"\n⚠️  Not all responses were 304: {etag_statuses}")

if __name__ == '__main__':
    main()
