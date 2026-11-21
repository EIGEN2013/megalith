# Push committed work to the GitHub repo (uses the gh login of this terminal).
$env:PATH = "$env:LOCALAPPDATA\Programs\MinGit\cmd;$env:LOCALAPPDATA\Programs\gh\bin;$env:PATH"
Set-Location (Split-Path $PSScriptRoot -Parent)
git push origin main
git log --oneline -1
Write-Host "PUSH DONE"
