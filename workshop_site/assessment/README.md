# Anonymous workshop assessment

This directory contains the mobile-first pre/post assessment for **SBI for experimental evolution**. It is a static application served by GitHub Pages; anonymous events are appended to a private Google Sheet through the small bound Apps Script in `google-sheets/Code.gs`.

The application supports:

- `assessment/?phase=pre`, returning to Chapter 1;
- `assessment/?phase=post`, returning to the workshop home page;
- six versioned knowledge activities, two repeated confidence items, and two post-only evaluation items;
- pointer drag, tap-to-select/tap-to-place, and keyboard-operable buttons;
- anonymous same-device pairing and an optional cross-device pairing code;
- append-only `started` and `completed` events, duplicate-safe event UUIDs, and an offline retry queue.

## Data collected

Each Sheet row is an event. The bound Apps Script creates `received_at` as the canonical UTC server timestamp.

| Field | Purpose |
| --- | --- |
| `id` | Random event UUID; unique key and retry deduplication |
| `received_at` | Server-generated event time |
| `run_id` | Connects `started` and `completed` for one opted-in run |
| `participant_id` | Random UUID used for same-device pre/post pairing |
| `phase` | `pre` or `post` |
| `event_type` | `started` or `completed` |
| `venue` | `NYU`, `UMN`, `TAU`, or `ONLINE` |
| `assessment_version` | Released questionnaire version |
| `workshop_git_sha` | Exact deployed workshop revision |
| `consent_version` | Notice version accepted by the participant |
| `payload` | Version hashes, raw answers, variants/orders, and client durations |

Every raw answer contains `question_id`, `question_revision`, `variant_id`, `displayed_order`, `response`, and `duration_ms`. The completed event also stores `client_duration_ms`. Per-question timing is intended to identify confusing interactions, not to monitor people.

## Data intentionally not collected

The assessment does not collect names, email addresses, phone numbers, university usernames, departments, labs, age, gender, career stage, precise geolocation, coordinates, IP addresses as study variables, browser user agents, screen dimensions, referrers, timezone, advertising identifiers, analytics identifiers, or free text. It uses no cookies, analytics, fingerprinting, session replay, or advertising scripts.

GitHub Pages, Google Apps Script, and other hosting/network providers can receive ordinary network metadata such as IP addresses in operational logs. Those provider logs are **not part of this research dataset** and must not be joined to or imported into assessment exports. Configure provider log retention according to the applicable institutional policy.

The notice is versioned centrally in `config.js` as `consentVersion` and `consent`. Its wording can be updated after ethics review by releasing a new consent version. The repository does not claim that ethics approval has been obtained.

## Anonymous pairing

After opt-in, the browser creates a random UUID with `crypto.randomUUID()` and stores it in `localStorage` under `evoSbiWorkshopParticipantId`. The UUID encodes no venue, date, identity, or device information. The same browser sends that UUID in the pre and post events. No cookie is used.

The pre completion also displays a cryptographically randomized code such as `WOLF-K7PM-4Q2X`. A participant may enter it on a second device. That device still receives a fresh random participant UUID; the offline scorer pairs the two records by the voluntarily entered anonymous code. A post response can always continue without a code and remains an unpaired observation.

## Connect your private Google Sheet once

No SQL or database console is required.

1. Create a blank Google Sheet in the Google account that should own the responses. Keep its sharing setting **Restricted**.
2. In that Sheet, open **Extensions → Apps Script**.
3. Replace the editor contents with `google-sheets/Code.gs`, save, then choose **Deploy → New deployment → Web app**.
4. Set **Execute as: Me** and **Who has access: Anyone**. Authorize the script and copy the deployed URL ending in `/exec`.
5. Paste that URL into `googleSheetsEndpoint` in `config.js`, rebuild, commit, and push. The current deployment is configured this way.

The script automatically creates an `assessment_events` tab with readable columns. It executes with the Sheet owner’s permissions, so participants can append validated events without receiving permission to open the private spreadsheet. Visiting the endpoint returns only a health message—never response data. Event UUIDs are checked before append, so retries do not duplicate rows.

If the Sheet endpoint is not configured or is temporarily unavailable, completed events remain in the browser’s local retry queue. The participant sees the exact state of the submission. The queue is retried when the browser next loads the assessment or returns online. Clearing site storage before a retry will remove that local copy.

Local previews served from `localhost`, `127.0.0.1`, or `::1` never transmit to the production Sheet. This keeps automated tests and presenter rehearsals out of the research dataset; persistence is enabled only on the deployed site.

## Export and score an event

Open the private Google Sheet and select the `assessment_events` tab. Each completed event has one row, with the six knowledge responses, confidence ratings, and post-only ratings repeated in readable columns after the authoritative `payload` column. Filter `event_type` to `completed` to inspect finished assessments.

To aggregate the data, use **File → Download → Comma-separated values (.csv)** while that tab is active. Export only the rows required for the workshop being analyzed and keep raw exports in approved research storage, not in Git.

Run the aggregate scorer locally:

```bash
python workshop_site/assessment/analysis/score_assessment.py assessment_events.csv
python workshop_site/assessment/analysis/score_assessment.py assessment_events.csv --group-by-venue
```

It accepts the Google Sheet CSV, a JSON array, or JSONL. By default it prints aggregates only: pre/post N, matched N, means/medians, within-person change, item-level correct percentages, and confidence means. It never prints participant rows. Raw answers remain authoritative; the browser’s post score is only friendly feedback.

For multiple deployed assessment versions, run the matching immutable question bank separately. The event’s `assessment_version` and `question_bank_hash` identify it.

## Versioning and reproducibility

`questions/v1.0.0.json` is the original released bank. `questions/v1.1.0.json` introduces parallel pre/post forms and is the active released bank. Do not silently edit either file. Any change to wording, scoring, choices, variants, or illustrations requires:

1. copy the active file to a new semantic version, for example `questions/v1.2.0.json`;
2. update `assessmentVersion` and `questionBankPath` in `config.js`;
3. update the scorer default or explicitly supply `--question-bank`;
4. rebuild and run the checks;
5. commit the old and new banks so historical responses remain reproducible.

`build_site.py` writes `build-meta.json` from the exact GitHub Actions commit SHA, build time, assessment version, and SHA-256 question-bank hash. Local builds use the current Git commit. The completed payload repeats those values.

## QR codes

The build generates three SVG QR codes from `PUBLIC_URL` in `build_site.py`:

- `assets/assessment-pre-qr.svg`;
- `assets/assessment-post-qr.svg`;
- `assets/workshop-qr.svg`.

After changing the public URL, regenerate them with:

```bash
cd tutorials
python workshop_site/build_site.py
```

The home-page primary QR targets the pre assessment. The Chapter 2 final card targets the post assessment. The second home-page QR preserves a direct route to the workshop website.

## University marks and original art

The university marks are decorative and every venue card also has a text label:

- `assets/nyu-official-seal.svg`: New York University seal, obtained through the Wikimedia Commons file mirror of the official mark;
- `assets/umn-official-logo.svg`: University of Minnesota official logo, obtained through the Wikimedia Commons file mirror;
- `assets/tau-official-logo.png`: downloaded from `english.tau.ac.il`, the university’s official English site.

`assets/lone-wolf.svg` is original artwork created for this workshop. Do not substitute redrawn university trademarks; replace a university file only with an authentic institutional asset and preserve the descriptive text label.

## Checks

Run:

```bash
python workshop_site/build_site.py
python workshop_site/tests/run_checks.py
python workshop_site/tests/visual_check.py
```

The reproducible checklist in `manual-test-checklist.md` covers Sheet privacy, mobile interactions, offline recovery, JavaScript-disabled behavior, and accessibility/Lighthouse review.
