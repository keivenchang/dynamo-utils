#!/usr/bin/python3
import subprocess
import smtplib
import socket
import time
import urllib.request
import os.path
import dns.query
import dns.tsigkeyring
import dns.update
import sys

debug = 0

# The hostname to update. Will resolve to <host>.dyn.nvidia.com
hostname = subprocess.check_output('hostname', shell=True).decode("utf-8").strip()
#hostname = 'jensen'
active_nic = os.popen("ip route | awk '/^default/ && ! /vpn0/ && ! /tun0/ && ! /cscotun0/ { print $5 }' | sort -u | head -1").read().strip() 
active_ipv4 = os.popen('ip addr show {0}'.format(active_nic)).read().split("inet ")[1].split("/")[0]
active_vpn_nic = os.popen("ip addr | awk '/<POINTOPOINT/ && /UP/ {print $2}' | sed 's/.$//'").read().strip()
active_vpn_ipv4 = os.popen('ip addr show {0}'.format(active_vpn_nic)).read().split("inet ")[1].split("/")[0]
check_corp_ping = subprocess.Popen('ping -c 1 -I {0} 10.120.237.52'.format(active_ipv4), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
check_corp_ping.wait()
check_corp_ping_vpn = subprocess.Popen('ping -c 1 -I {0} 10.120.237.52'.format(active_vpn_ipv4), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
check_corp_ping_vpn.wait()
#check_ddns_ping = subprocess.Popen('ping -c 1 $(hostname).dyn.nvidia.com', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#check_ddns_ping.wait()

# Set to 1 to update dns records upon change
updatedns = 1

# Init variables
changedetected = 0

def dnsupdate ():
	if debug == 1: 
		print('INFO: Doing DNS update')
	if debug == 1: 
		print('INFO: ip is', active_ipv4)
	update = dns.update.Update('dyn.nvidia.com')
	update.replace(hostname, 60, 'a', active_ipv4)
	response = dns.query.tcp(update, '10.120.237.52')
	if debug == 1: 
		print('INFO: response ID is', str(response).replace("id", "-"))

def dnsupdate_vpn ():
    if debug == 1:
        print('INFO: Doing DNS update')
    if debug == 1:
        print('INFO: ip is', active_vpn_ipv4)
    update = dns.update.Update('dyn.nvidia.com')
    update.replace(hostname, 60, 'a', active_vpn_ipv4)
    response = dns.query.tcp(update, '10.120.237.52')
    if debug == 1:
        print('INFO: response ID is', str(response).replace("id", "-"))


if check_corp_ping.returncode == 0:
    dnsupdate()
    print("DDNS is configured!")
    print("DDNS hostname is:", hostname + ".dyn.nvidia.com")
elif check_corp_ping_vpn.returncode == 0:
    dnsupdate_vpn()
    print("DDNS is configured!")
    print("DDNS hostname is:", hostname + ".dyn.nvidia.com")
else:
    print("No corp connection. Exitig...")
    exit()
