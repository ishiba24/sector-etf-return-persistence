# Canvas submission checklist (QM Final Project)

## Before you paste the GitHub URL

1. **Create the GitHub repository** (github.com → New repository). Suggested name: `qm-final-project` or `quant-mentorship-final-austin-renz`.
2. **Confirm `data/etf_prices.csv` is pushed** (GitHub web UI → `data/` folder). Graders will not run download APIs.
3. **Set your Git author** (optional but professional), then amend the last commit if needed:

   ```bash
   git config user.name "Austin Renz"
   git config user.email "YOUR_EMAIL@gatech.edu"
   git commit --amend --reset-author --no-edit
   ```

4. **Push:**

   ```bash
   cd /path/to/Quant-Final
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

5. **Canvas:** QM Week 13 → **QM Final Project** → submit the repo link, e.g. `https://github.com/YOUR_USERNAME/YOUR_REPO`.

## Research citations

Listed in **`REFERENCES.md`** (also summarized in the PDF/notebook where relevant).
