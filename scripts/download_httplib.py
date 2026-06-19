#!/usr/bin/env python3
"""Download httplib.h from GitHub"""
import urllib.request

url = 'https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h'
data = urllib.request.urlopen(url).read()
with open('qwen3-asr.cpp/third_party/httplib.h', 'wb') as f:
    f.write(data)
print(f'Downloaded {len(data)} bytes')
