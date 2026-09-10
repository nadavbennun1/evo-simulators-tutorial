# Assessment release checklist

Record the date, commit SHA, browser versions, device/viewport, tester, and result with every release.

## Static and navigation

- [ ] Home-page primary QR decodes to `/assessment/?phase=pre`.
- [ ] The bottom home-page QR decodes to the workshop home page.
- [ ] Pre completion opens Chapter 1; post completion opens the workshop home page.
- [ ] The Chapter 2 final card button and QR both open `?phase=post`.
- [ ] With JavaScript disabled, the assessment explains the limitation and links to the workshop.

## Interaction and mobile

- [ ] At 390 px width, no page or question has horizontal overflow.
- [ ] NYU, UMN, TAU, and Online cards show an image plus a readable text label.
- [ ] Complete all six activities with touch/pointer dragging.
- [ ] Complete all matching/ordering/flow activities using tap-select then tap-place.
- [ ] Complete the full assessment using only keyboard, including focus visibility.
- [ ] Check VoiceOver or NVDA labels for venue cards, slots, plots, scales, progress, and navigation.
- [ ] With reduced motion enabled, content remains understandable and animations are suppressed.

## Study behavior

- [ ] Declining transmits nothing and routes directly to the workshop.
- [ ] `started` is created only after opt-in.
- [ ] Pre reveals neither correctness nor a numerical score.
- [ ] Post feedback and score appear only after the completed event has been queued/submitted.
- [ ] A same-browser pre/post pair uses the same participant UUID.
- [ ] A second browser can enter the pre pairing code; continuing unpaired also works.
- [ ] Raw answer records contain ID, revision, variant, displayed order, response, and duration.
- [ ] Payload inspection confirms that no prohibited personal/browser fields are present.
- [ ] `build-meta.json` values appear in the completed event and its question-bank hash verifies.

## Persistence and permissions

- [ ] Simulate offline mode, complete the assessment, and confirm the local-save message.
- [ ] Restore connectivity/revisit and confirm the queued event is inserted.
- [ ] Retry an identical event UUID and confirm there is only one database row.
- [ ] With the public anon key, `INSERT` succeeds and `SELECT`, `UPDATE`, and `DELETE` fail.
- [ ] Started and completed rows have database-created `received_at` timestamps.

## Regression and accessibility

- [ ] Run `python workshop_site/tests/run_checks.py`.
- [ ] Run `python workshop_site/tests/visual_check.py`.
- [ ] Exercise the existing Chapter 1 and Chapter 2 interactives and global reset behavior.
- [ ] Run Lighthouse accessibility checks on home, pre, and post pages at mobile width; resolve critical findings before release.
