#!/usr/bin/env python3
"""
Helper script to push this project to GitHub.
Usage:
    python upload_to_github.py <github_username> <github_token> <repo_name>

Example:
    python upload_to_github.py AqeelAhmed mytoken123 wine-classification
"""
import sys, subprocess

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False
    print(f"  OK: {result.stdout.strip()}" if result.stdout.strip() else "  OK")
    return True

if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)

username, token, repo = sys.argv[1], sys.argv[2], sys.argv[3]
remote = f"https://{username}:{token}@github.com/{username}/{repo}.git"

print(f"\n→ Initialising git …")
run("git init")
run("git config user.email 'f22bseen1e02085@student.edu'")
run(f"git config user.name '{username}'")

print("→ Staging files …")
run("git add .")
run("git commit -m 'Initial commit: Wine Quality Classification ML project'")

print(f"→ Pushing to github.com/{username}/{repo} …")
run(f"git remote add origin {remote} 2>/dev/null || git remote set-url origin {remote}")
run("git branch -M main")
if run(f"git push -u origin main"):
    print(f"\n✅ Done! Repository URL:\n   https://github.com/{username}/{repo}")
else:
    print("\n❌ Push failed. Make sure the repo exists on GitHub and the token is valid.")
