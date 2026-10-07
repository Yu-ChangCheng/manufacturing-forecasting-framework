# Publish this example on GitHub

This repository is designed to be public. The two original company notebooks and all NIW, H-1B, sales, and customer records must stay outside this repository.

## Before publishing

1. Review every file in this folder. The intended public files are the Python package, local Streamlit app and launcher, output-free technical notebook, tests, one fictional CSV, attributed public car-parts subset, generator scripts, documentation, license, and project metadata.
2. Check for employer names, database hostnames, connection settings, passwords, customer names, and real sales figures. Do not rely only on `.gitignore`; Git can include files that were added earlier.
3. Run `git status --short`, `git ls-files`, and `git log --oneline --all` inside this folder. Check the current files **and previous commits** before pushing: `git push` publishes the reachable commit history, not just today's folder contents.
4. Run the guided fictional demo, public-data example, and automated tests in the README. Backtest outputs and the downloaded full `.tsf` source should remain in the ignored `outputs/` folder.
5. If a real credential was ever included in a public commit, deleting the file later is insufficient; rotate the credential and remove it from Git history.

## Create the public repository

1. Sign in to the GitHub account `Yu-ChangCheng`.
2. Open [GitHub's new repository page](https://github.com/new). Set owner to `Yu-ChangCheng`, name to `manufacturing-forecasting-framework`, and visibility to **Public**. A useful description is: “Reproducible monthly demand forecasting with Random Forest, chronological backtests, transparent baselines, and a guided local app.” Add topics such as `demand-forecasting`, `random-forest`, `time-series`, `manufacturing`, and `streamlit` in the repository settings after creation. Leave GitHub's README, license, and `.gitignore` initialization boxes unchecked; those files already exist locally.
3. In PowerShell, from this repository folder, run:

```powershell
cd "C:\Users\leo.cheng\Downloads\NIW\manufacturing-forecasting-framework"
git status --short
git remote -v
git remote add origin https://github.com/Yu-ChangCheng/manufacturing-forecasting-framework.git
git remote -v
git push -u origin main
```

`git remote -v` currently returns nothing for this local repository. If it later shows an `origin`, inspect its URL and use `git remote set-url origin <correct-url>` only if needed; do not add a second `origin`. GitHub may open a browser or prompt you to sign in during the push. Use your own credentials in GitHub's authentication flow and never paste a token into this project or chat. These commands follow [GitHub's official guide for adding local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

4. Open the public repository in a signed-out or private browser window. Confirm that the README, equations, and diagrams render, the **Cite this repository** button appears from `CITATION.cff`, and the **Actions** tab shows the CI checks passing. GitHub documents [citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files). Inspect the public file list again. If the push is rejected because GitHub already initialized the remote with a README, stop and inspect the two histories; do not force-push blindly.
5. For a petition exhibit, save a dated PDF or screenshot of the public repository page, README, release or commit page, and any independent feedback. Describe the repository as an educational prototype. The public car-parts data and fictional sample scores are not company results or proof of adoption.

The app is configured for local computer use. Do not deploy it as a public CSV-upload service without a separate review of access control, retention, and data handling.

If the chosen repository name changes, update the URL in the commands, README, CI badge, and `CITATION.cff`.
