Danistay remote crawler node
============================
SETUP (one time, ~2 min):
  1. ./install_auth.sh <havelsan_user> <havelsan_pass>
       -> installs the link-keeper (cron) + stores creds safely (mode 600)
  2. ./start.sh 2011 1983
       -> pip deps + crawler + publisher, in background

That's it. It crawls its year range and publishes every new decision to the
central pipeline (10.150.41.5). Dedup is shared; docs appear in the central
web UI automatically. No other setup.

DAILY OPS (nothing needed):
  Progress:  tail logs/crawl_range.log
  Published: tail logs/publish.log
  Net link:  tail ~/.config/havelsan-login.log
  Stop:      pkill -f crawl_range; pkill -f publish_loop
  Resume:    ./start.sh 2011 1983   (both halves are resumable)
