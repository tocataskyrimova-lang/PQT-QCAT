# Restore

Clone GitHub, checkout the annotated freeze tag and download release assets. Verify every asset against SHA256SUMS.txt. Extract ARTIFACTS.zip to recover the original local archive filenames. The public archive deduplicates identical ZIP payloads and preserves original ZIP SHA-256. Follow the per-experiment reports and restore/verify commands in the individual delivery archives. Original NPZ snapshots preserve agent state including K/Z, configuration and RNG where recorded. Check CHECKPOINT_MANIFEST.json for actual snapshot inventory. No missing fields should be synthesized.

Use the release Git bundle to recover historical branches if needed. The bundle excludes the new archive commit and preserves all inventoried historical refs. Source trees and all historical scientific results remain immutable.

Independent F0B verification uses scripts/verify_f0b_coverage.py against the unpacked F0 query table. Exact-resume tests are test_agent_exact_resume.py and checkpoint/online tests retained in the source. These are restoration checks, not new experimental panels.
