# Commit everything (per .gitignore) and push to a public GitHub repo "megalith".
$env:PATH = "$env:LOCALAPPDATA\Programs\MinGit\cmd;$env:LOCALAPPDATA\Programs\gh\bin;$env:PATH"
Set-Location (Split-Path $PSScriptRoot -Parent)
$u = gh api user | ConvertFrom-Json
Write-Host "GitHub user: $($u.login)"
if (-not (Test-Path .git)) { git init -b main | Out-Null }
git config user.name $u.login
git config user.email "$($u.id)+$($u.login)@users.noreply.github.com"
git config core.longpaths true
git config core.autocrlf false
gh auth setup-git
git add -A
git commit -q -m "Megalith: web game, Blender/Python asset pipeline, photoscanned nature, castle, monsters"
git log --oneline -1
gh repo create megalith --public --source . --push --description "Megalith - browser 3D action adventure (three.js) with a Blender/Python asset pipeline"
Write-Host "DONE: https://github.com/$($u.login)/megalith"
