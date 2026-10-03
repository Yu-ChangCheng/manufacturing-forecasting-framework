# Publish this example on GitHub

This repository is designed to be public. The two original company notebooks and all NIW, H-1B, sales, and customer records must stay outside this repository.

## Before publishing

1. Review every file in this folder. The intended public files are the Python package, local Streamlit app and launcher, tests, two fictional CSVs, attributed public car-parts subset, generator and converter scripts, documentation, license, and project metadata.
2. Check for employer names, database hostnames, connection settings, passwords, customer names, and real sales figures. Do not rely only on `.gitignore`; Git can include files that were added earlier.
3. Run `git status --short` and `git ls-files` inside this folder. Check the full list before pushing.
4. Run the guided fictional demo, public-data example, and automated tests in the README. Backtest outputs and the downloaded full `.tsf` source should remain in the ignored `outputs/` folder.
5. If a real credential was ever included in a public commit, deleting the file later is insufficient; rotate the credential and remove it from Git history.

## Create the public repository

1. Sign in to the GitHub account `Yu-ChangCheng`.
2. Choose **New repository**. Name it `manufacturing-forecasting-framework`, set visibility to **Public**, and leave GitHub's README, license, and `.gitignore` initialization boxes unchecked.
3. In PowerShell, from this folder, run:

```powershell
git remote add origin https://github.com/Yu-ChangCheng/manufacturing-forecasting-framework.git
git push -u origin main
```

4. Open the public repository in a signed-out or private browser window. Confirm that the README and diagram render and that only the intended files appear.
5. For a petition exhibit, save a dated PDF or screenshot of the public repository page, README, release or commit page, and any independent feedback. Describe the repository as an educational prototype. The public car-parts data and fictional sample scores are not company results or proof of adoption.

The app is configured for local computer use. Do not deploy it as a public CSV-upload service without a separate review of access control, retention, and data handling.

GitHub may ask you to authenticate during `git push`. If the chosen repository name changes, replace the URL in the command and in the README.
