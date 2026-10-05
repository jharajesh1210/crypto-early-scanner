name: Crypto 5M Buy Sell Scanner

on:
  workflow_dispatch:

  schedule:
    - cron: "*/5 * * * *"

permissions:
  contents: write

jobs:
  run-scanner:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy requests

      - name: Run Crypto 5M Scanner
        env:
          BOT_TOKEN: ${{ secrets.BOT_TOKEN }}
          CHAT_ID: ${{ secrets.CHAT_ID }}
        run: |
          python crypto_early_alert.py

      - name: Save scanner data
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"

          git add "*.csv" "*.json" 2>/dev/null || true

          if ! git diff --cached --quiet; then
            git commit -m "Update crypto scanner data"
            git push
          else
            echo "No scanner data changes to save."
          fi
