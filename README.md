# Danistay remote crawler node

Crawls its year range from karararama.danistay.gov.tr and publishes every new
decision to the central pipeline (10.150.41.5). Dedup is shared; docs appear
in the central web UI automatically.

## Setup (one time, ~2 min)

```bash
git clone https://github.com/lilofthea/danistay_crawler.git
cd danistay_crawler

# 1) Havelsan link-keeper (cron) — replaces <user> and <pass> with the
#    real havelsan username and password, no brackets:
./install_auth.sh jdoe  SuperSecret123

# 2) Start crawler + publisher in the background (year range optional,
#    default 2011 -> 1983):
./start.sh 2011 1983
```

That's the whole setup. Both processes run unattended (`nohup`), the crawler
is resumable and the publisher is incremental + deduplicated.

## Daily ops (nothing needed, just checks)

```bash
tail logs/crawl_range.log        # crawl progress
tail logs/publish.log            # what got published to the pipeline
tail ~/.config/havelsan-login.log  # network link state
```

Stop:  `pkill -f crawl_range; pkill -f publish_loop`
Resume: `./start.sh 2011 1983`  (both halves pick up where they left off)

## Notes

- If havelsan re-auth ever logs "token expired", open the portal page in a
  browser once, copy the fresh token URL, and update PORTAL_URL in
  `auth/havelsan-login.sh` (then re-run `./install_auth.sh ...` to reinstall).
- Credentials live only in `~/.config/havelsan-creds` (mode 600), never in
  the repo or the scripts.
