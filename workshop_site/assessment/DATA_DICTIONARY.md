# Assessment data dictionary and privacy boundary

The unit of storage is an append-only event. A consenting run normally produces one `started` row and one `completed` row in a private Google Sheet. `id` is the immutable submission/event key; `run_id` joins those two rows. The bound Apps Script assigns `received_at` as a canonical UTC ISO timestamp.

## Database columns

| Column | Type | Allowed values / meaning |
| --- | --- | --- |
| `id` | UUID | Cryptographically random client event ID; primary key |
| `received_at` | UTC ISO string | Server-created time from the bound Apps Script |
| `run_id` | UUID | Random assessment-run ID |
| `participant_id` | UUID | Random anonymous same-device pairing ID |
| `phase` | TEXT | `pre`, `post` |
| `event_type` | TEXT | `started`, `completed` |
| `venue` | TEXT | `NYU`, `UMN`, `TAU`, `ONLINE` |
| `assessment_version` | TEXT | Semantic version of the immutable bank |
| `workshop_git_sha` | TEXT | 40-character deployed Git SHA (`development` only in local builds) |
| `consent_version` | TEXT | Version of the notice shown before opt-in |
| `payload` | JSON string | Event-specific values below; maximum 64 KiB |

## Completed payload

| JSON path | Type | Meaning |
| --- | --- | --- |
| `assessment_schema_version` | string | Payload contract version |
| `assessment_version` | string | Questionnaire release |
| `question_bank_hash` | string | SHA-256 of the exact question-bank JSON |
| `workshop_git_sha` | string | Repeated deployment revision for portable exports |
| `consent_version` | string | Repeated notice version |
| `phase` | string | Repeated pre/post phase |
| `venue` | string | Repeated venue |
| `pairing_code` | string or null | Optional random cross-device code |
| `client_duration_ms` | integer | Total client elapsed time after opt-in |
| `answers.knowledge` | array | Six authoritative raw knowledge answers |
| `answers.confidence` | array | Two 1–5 ratings, never part of knowledge score |
| `answers.post_evaluation` | array | Two post-only 1–5 ratings; empty for pre |

Every answer object contains `question_id`, `question_revision`, `variant_id`, `displayed_order`, `response`, and `duration_ms`. Responses are IDs, arrays of IDs, mappings between IDs, or integers; no free text is accepted.

For convenience, the Sheet repeats each response and total client duration in clearly named columns to the right of `payload`. These are display columns derived from the authoritative payload, not additional measurements. The scorer always uses `payload`.

## Explicit privacy boundary

No field is defined for identity, contact information, demographics, laboratory/department, precise position, network address, browser/device fingerprint, referrer, timezone, analytics, advertising identifiers, or participant-authored text. Do not extend exports with hosting-provider logs. If a future project needs any additional field, release a new schema and consent notice and obtain the required institutional review before collection.
