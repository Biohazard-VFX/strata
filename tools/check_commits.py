#!/usr/bin/env python3
"""Validate Conventional Commits headers, separation and breaking-change footers."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

HEADER = re.compile(r"[A-Za-z][A-Za-z0-9_-]*(?:\([^\s()](?:[^()\r\n]*[^\s()])?\))?!?: \S[^\r\n]*")
BREAKING = re.compile(r"BREAKING(?: CHANGE|-CHANGE): \S.*")


def validate(message):
    lines = message.replace('\r\n','\n').rstrip('\n').split('\n')
    if not lines or not HEADER.fullmatch(lines[0]):
        return 'expected type[(scope)][!]: description'
    if len(lines) > 1 and lines[1]:
        return 'body or footers must follow a blank line'
    for line in lines[2:]:
        if line.startswith(('BREAKING CHANGE:','BREAKING-CHANGE:')) and not BREAKING.fullmatch(line):
            return 'breaking-change footer requires a colon, space and description'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('range',nargs='?',default='origin/main..HEAD')
    parser.add_argument('--message-file',type=Path)
    args = parser.parse_args()
    if args.message_file:
        error = validate(args.message_file.read_text())
        if error:
            sys.exit(f'commit message: {error}')
        return
    revisions = subprocess.check_output(['git','rev-list','--reverse',args.range],text=True).splitlines()
    for revision in revisions:
        message = subprocess.check_output(['git','show','-s','--format=%B',revision],text=True)
        error = validate(message)
        if error:
            sys.exit(f'{revision}: {error}: {message.splitlines()[0]}')
    print(f'Conventional Commits: {len(revisions)} commit(s) checked in {args.range}')


if __name__ == '__main__':
    main()
