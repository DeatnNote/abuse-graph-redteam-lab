# Results

3 independent simulated worlds, each with about 23,826 advertiser accounts (20,000 independent advertisers, 60 benign agencies, 60 fraud rings from 25 operators). Detectors train on half the rings and agencies and are tested on the other half. Mean (min–max) over worlds. All data simulated.

Each test ring is then attacked by the automated red team, which can buy evasion actions for each of its accounts (prices in `agl/redteam.py`) and query the detector's risk score. A ring **evades** when fewer than half of its accounts are flagged. Threshold: 1% false positives on benign training accounts.

## Detection before and after attack

| Detector | Ring recall (no attack) | False-positive rate | Agency false-positive rate | Rings evading at ≤$10/account | at ≤$25 | at ≤$50 | at ≤$100 | Median cost to evade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0: account features only | 40% (28%–52%) | 1.0% (0.9%–1.1%) | 0.0% (0.0%–0.0%) | 100% (100%–100%) | 100% (100%–100%) | 100% (100%–100%) | 100% (100%–100%) | $1 ($0–$3) |
| v1: + entity resolution & graph features | 100% (100%–100%) | 0.2% (0.0%–0.7%) | 1.9% (0.0%–5.3%) | 0% (0%–0%) | 22% (7%–50%) | 98% (97%–100%) | 99% (97%–100%) | $27 ($23–$28) |
| v1 without e-mail entity resolution | 100% (100%–100%) | 1.0% (0.0%–2.2%) | 7.3% (0.0%–15.0%) | 2% (0%–3%) | 51% (7%–90%) | 93% (87%–100%) | 99% (97%–100%) | $20 ($13–$28) |
| v2 naive: + prepaid-card & residential-IP signals | 100% (100%–100%) | 0.4% (0.1%–0.6%) | 2.2% (0.5%–4.1%) | 92% (76%–100%) | 100% (100%–100%) | 100% (100%–100%) | 100% (100%–100%) | $5 ($5–$5) |
| v2 hardening round 1 (prepaid-card signal + adversarial examples) | 100% (100%–100%) | 0.5% (0.1%–0.7%) | 1.7% (0.5%–4.1%) | 71% (27%–100%) | 100% (100%–100%) | 100% (100%–100%) | 100% (100%–100%) | $8 ($5–$15) |
| v2 hardening round 2 | 100% (100%–100%) | 0.4% (0.1%–0.5%) | 1.5% (0.1%–4.1%) | 1% (0%–3%) | 4% (0%–9%) | 32% (0%–69%) | 57% (7%–84%) | $74 ($28–$115) |
| v2 hardening round 3 | 100% (100%–100%) | 0.6% (0.0%–1.2%) | 4.7% (0.0%–9.7%) | 1% (0%–3%) | 1% (0%–3%) | 6% (0%–19%) | 19% (3%–31%) | $120 ($90–$146) |

## What the red team bought

| Detector | Evasion actions chosen (rings, summed over worlds) |
|---|---|
| v0: account features only | fresh_emails (91) |
| v1: + entity resolution & graph features | fresh_emails (91), paraphrase_ads (90), prepaid_cards (89), new_devices (84), aged_domains (2), agency_camouflage (1) |
| v2 naive: + prepaid-card & residential-IP signals | residential_ips (91), prepaid_cards (19), paraphrase_ads (6) |
| v2 hardening round 1 (prepaid-card signal + adversarial examples) | residential_ips (91), paraphrase_ads (50), prepaid_cards (49), fresh_emails (25), new_devices (3), credit_cards (1) |
| v2 hardening round 3 | fresh_emails (84), paraphrase_ads (73), new_devices (67), credit_cards (65), agency_camouflage (61), aged_domains (49), residential_ips (48), stagger_creation (33), prepaid_cards (26), email_variants (25), throttle_spend (12) |

## Findings

* **Graph features catch every unmodified ring**, against 28-52% of rings for account features alone, and account-only detection is evaded for free or for a few dollars of new e-mail addresses.
* **But v1 is cheap to beat**: fresh e-mails, paraphrased ads, prepaid cards and emulated devices break every link the graph relies on for roughly $20-30 per account.
* **One round of hardening only moved the weak spot**: after training on the v1 attack variants, 92% of rings evaded the naive v2 and 71% evaded round 1 for $10 per account or less, mostly with a $5 residential proxy. The model apparently shifted its weight onto the IP block, which the first attack never changed.
* **Iterated adversarial training raised the price**: after three rounds of attacking the training rings and retraining, 16-97% of test rings (depending on the world) could not evade at all with the full action menu, and those that did paid $90-146 per account instead of $23-28. Hardening is not free: false positives on benign agencies rose to 4-10% in two of the three worlds.
* **E-mail entity resolution** raised the median cost to evade in two of three worlds ($18 vs $23, $13 vs $28) and cut false positives on agencies in one (15% to 0.3%); a determined ring simply buys unrelated e-mails.

## Distributed entity resolution (PySpark)

Connected components over the account-attribute graph by alternating min-label propagation; checked against scipy.

| Accounts | Clusters | Iterations | Same clusters as scipy | Spark seconds (local, 2 cores) | scipy seconds |
|---:|---:|---:|:---:|---:|---:|
| 24,175 | 19,848 | 6 | yes | 21.6 | 0.01 |
| 239,371 | 198,651 | 6 | yes | 58.9 | 0.13 |

On one machine scipy is far faster; the Spark job is the version that scales past one machine's memory.

