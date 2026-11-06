# Import the LOMO addendum using Windows PowerShell

The ZIP `HPI_LOMO_GitHub_ready.zip` contains only new files and folders, not replacement versions of previous HPI or OOF code and inputs. The original repository's root `.gitignore` ignores `results/` directories broadly, so this addendum includes `outputs/lomo/.gitignore` to explicitly include the archived LOMO results while ignoring regeneration folders.

**Prerequisites:** Git for Windows, write access to `arikampa/hpi_citation_sentiment_analysis` and a local checkout (the commands below create one if missing). Download `HPI_LOMO_GitHub_ready.zip` to your Downloads folder. If you keep the repository elsewhere, edit the `$Repo` path before running.

```powershell
$Repo = Join-Path $HOME 'hpi_citation_sentiment_analysis'
$Zip = Join-Path $HOME 'Downloads\HPI_LOMO_GitHub_ready.zip'
$Stage = Join-Path $env:TEMP 'hpi_lomo_addendum_20261106'

if (-not (Test-Path -LiteralPath $Zip)) { throw "ZIP not found: $Zip" }
if (-not (Test-Path -LiteralPath (Join-Path $Repo '.git'))) {
    git clone https://github.com/arikampa/hpi_citation_sentiment_analysis.git $Repo
}
Set-Location $Repo
if (git status --porcelain) { throw 'Working tree has changes. Commit/stash them before importing LOMO.' }
git pull --ff-only origin main

if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
Expand-Archive -LiteralPath $Zip -DestinationPath $Stage

foreach ($rel in @('docs\lomo', 'notebooks\lomo', 'scripts\lomo', 'outputs\lomo', 'LOMO_README.md')) {
    if (Test-Path -LiteralPath (Join-Path $Repo $rel)) { throw "Already exists in checkout: $rel" }
}
foreach ($top in @('docs', 'notebooks', 'scripts', 'outputs')) {
    Copy-Item -LiteralPath (Join-Path $Stage "$top\lomo") -Destination (Join-Path $Repo $top) -Recurse
}
Copy-Item -LiteralPath (Join-Path $Stage 'LOMO_README.md') -Destination $Repo

git status --short
git add -- LOMO_README.md docs/lomo notebooks/lomo scripts/lomo outputs/lomo
git diff --cached --stat
```

**Review** the displayed added files. If satisfied, publish with:

```powershell
git commit -m "Add reproducible repeated 20x5 LOMO GHPI sensitivity analysis"
git push origin main
```

Optional numerical rerun from the repository root, if a compatible Python environment is installed:

```powershell
python -m pip install -r requirements-analysis.txt
python scripts/lomo/LOMO_repeated20x5.py
python scripts/lomo/verify_lomo_outputs.py
```

If you have not installed Git for Windows, install it first. If `git push` requests authentication, complete the login with your GitHub account; do not paste tokens into source files.
