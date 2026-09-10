# Anonymous workshop assessment

This directory contains the mobile-first pre/post assessment for **SBI for experimental evolution**. It is a static application served by GitHub Pages; anonymous events are sent to a separate Supabase/Postgres project through the isolated adapter in `backend.js`.

The application supports:

- `assessment/?phase=pre`, returning to Chapter 1;
- `assessment/?phase=post`, returning to the workshop home page;
- six versioned knowledge activities, two repeated confidence items, and two post-only evaluation items;
- pointer drag, tap-to-select/tap-to-place, and keyboard-operable buttons;
- anonymous same-device pairing and an optional cross-device pairing code;
- append-only `started` and `completed` events, duplicate-safe event UUIDs, and an offline retry queue.

## Data collected

Each database event contains the fields below. PostgreSQL creates `received_at`; it is the canonical UTC-capable server timestamp.

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

GitHub Pages, Supabase, and other hosting/network providers can receive ordinary network metadata such as IP addresses in operational logs. Those provider logs are **not part of this research dataset** and must not be joined to or imported into assessment exports. Configure provider log retention according to the applicable institutional policy.

The notice is versioned centrally in `config.js` as `consentVersion` and `consent`. Its wording can be updated after ethics review by releasing a new consent version. The repository does not claim that ethics approval has been obtained.

## Anonymous pairing

After opt-in, the browser creates a random UUID with `crypto.randomUUID()` and stores it in `localStorage` under `evoSbiWorkshopParticipantId`. The UUID encodes no venue, date, identity, or device information. The same browser sends that UUID in the pre and post events. No cookie is used.

The pre completion also displays a cryptographically randomized code such as `WOLF-K7PM-4Q2X`. A participant may enter it on a second device. That device still receives a fresh random participant UUID; the offline scorer pairs the two records by the voluntarily entered anonymous code. A post response can always continue without a code and remains an unpaired observation.

## Configure Supabase once

1. Create a dedicated Supabase project in the institutionally appropriate region and account.
2. Run `supabase/001_assessment_events.sql` in its SQL editor.
3. In `config.js`, set `supabaseUrl` and the **public anon key**. Never place a service-role key in this repository or browser code.
4. Confirm with a disposable project that an anonymous REST request can `INSERT`, while anonymous `SELECT`, `UPDATE`, and `DELETE` requests are denied.
5. Rebuild, test, commit, and deploy.

The migration revokes every table privilege from `anon`, grants only `INSERT`, enables and forces row-level security, and supplies one insert policy. The primary key deduplicates network retries. A public anon key is safe here only while these restrictions remain in place.

If Supabase is not configured or is temporarily unavailable, completed events remain in the browser’s local retry queue. The participant sees the exact state of the submission. The queue is retried when the browser next loads the assessment or returns online. Clearing site storage before a retry will remove that local copy.

## Export and score an event

An authorized researcher can export `assessment_events` from the Supabase dashboard as CSV, or query it with a protected researcher/service connection. Export only the event and date range required for that workshop. Keep raw exports in approved research storage, not in Git.

Run the aggregate scorer locally:

```bash
python workshop_site/assessment/analysis/score_assessment.py assessment_events.csv
python workshop_site/assessment/analysis/score_assessment.py assessment_events.csv --group-by-venue
```

It accepts Supabase CSV, a JSON array, or JSONL. By default it prints aggregates only: pre/post N, matched N, means/medians, within-person change, item-level correct percentages, and confidence means. It never prints participant rows. Raw answers remain authoritative; the browser’s post score is only friendly feedback.

For multiple deployed assessment versions, run the matching immutable question bank separately. The event’s `assessment_version` and `question_bank_hash` identify it.

## Versioning and reproducibility

`questions/v1.0.0.json` is a released, immutable question bank. Do not silently edit it. Any change to wording, scoring, choices, variants, or illustrations requires:

1. copy the file to a new semantic version, for example `questions/v1.1.0.json`;
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

The reproducible checklist in `manual-test-checklist.md` covers database permissions, mobile interactions, offline recovery, JavaScript-disabled behavior, and accessibility/Lighthouse review.
