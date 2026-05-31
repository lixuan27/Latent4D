#!/bin/bash
# One-shot: commit, create the GitHub repo if needed, push, and enable GitHub Pages
# from docs/ on main. The token is passed as an argument and is NEVER written to
# any file or left in the git remote (scrubbed at the end).
#
# Usage:   bash push_and_pages.sh <GITHUB_TOKEN>
# Then rotate the token on GitHub, since it has been exposed in chat.
set -euo pipefail
TOKEN="${1:?pass your GitHub token as the first argument}"
OWNER=lixuan27
REPO=Latent4D
cd "$(dirname "$0")"

git init -q -b main 2>/dev/null || true
git config user.email "thutsyj23@gmail.com"
git config user.name  "$OWNER"

# Create the repo if it does not exist (ignore 'already exists').
curl -s -o /dev/null -w "create: %{http_code}\n" \
  -H "Authorization: token $TOKEN" \
  -d "{\"name\":\"$REPO\",\"description\":\"Latent-4D: 4D representation is already inside your 3D decoder. Project page + code.\",\"private\":false}" \
  https://api.github.com/user/repos || true

git add -A
git commit -q -m "Latent-4D project page and reproducible code" || echo "(nothing new to commit)"

git remote remove origin 2>/dev/null || true
git remote add origin "https://${TOKEN}@github.com/${OWNER}/${REPO}.git"
git push -u origin main

# Enable GitHub Pages from /docs on main.
curl -s -o /dev/null -w "pages: %{http_code}\n" -X POST \
  -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -d '{"source":{"branch":"main","path":"/docs"}}' \
  "https://api.github.com/repos/${OWNER}/${REPO}/pages" || true

# Scrub the token from the local git config.
git remote set-url origin "https://github.com/${OWNER}/${REPO}.git"
echo "Done. Page will be at https://${OWNER}.github.io/${REPO}/ within a minute."
echo "REMINDER: rotate the GitHub token now."
