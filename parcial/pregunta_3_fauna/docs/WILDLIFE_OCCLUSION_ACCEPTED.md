# Accepted occlusion review after human returns

The user explicitly directed the project to exclude the 18 cases without a level, select Miguel's review for case 121, and proceed with the notes as delivered. This exception applies to the independent 600-case review. The original V6 sources, model, predictions, full review, notebook, and draft PDF remain frozen.

The canonical sample CSV preserves all 600 original rows. The accepted-cohort export contains 582 rows: 340 low, 69 medium, 48 high, and 125 ambiguous. Case 121 retains Miguel's `medium` level, name, and empty note. No missing level or note is fabricated. The 18 excluded case numbers are 253, 302, 385-393, 409-414, and 439. There are 398 accepted cases without notes.

`review_acceptance.json` binds the decision to the five returned files, fixed sampling protocol, old canonical backup, excluded annotation IDs, and accepted CSV hashes. The original alternative review for case 121 remains in the archived source files. The older strict review workflow continues to require written evidence; the separate accepted workflow records this explicit waiver rather than claiming that evidence was supplied.

## Evaluation

Run `evaluate_occlusion_accepted.py check` with the frozen parent and sample output directories to verify the accepted inputs. Run its `run` action only once in the exact frozen V6 environment. It reuses matching completed levels and never trains or performs inference.

All trans_test frames remain in each low/medium/high evaluation, including empty frames and their false positives. Unreviewed, excluded, ambiguous, or other-level boxes are ignored with the unchanged COCO rules. Each evaluated level exceeds the original minimum support of 30 objects. AP is unweighted accepted-cohort performance. The confidence intervals are conditional 95% sequence-bootstrap intervals with 200 replicates and seed 42; they do not measure annotation disagreement, training-seed variation, or future-camera performance.

The 18 exclusions are human nonresponse. Descriptive proportions and retained design-weight ratios describe the accepted cases. Complete-600 population proportion intervals and the original planning guarantee are not reused. Population representativeness after exclusion is not asserted.

## Delivery and verification

The separate `build_occlusion_accepted_delivery.py` script waits for completed matching results through its input checks. Its `all` action prepares a portable evidence folder, executes the four notebook code cells in an isolated container, and generates a two-page PDF annex and numeric metrics CSV. A failed notebook prevents PDF publication. The annex is visually inspected after rendering; editorial approval remains pending.

The delivery supplements the original V6 notebook and ten-page PDF without changing them. Integrating the annex into the final report still requires editorial review within the assignment's page limit. Questions 1 and 2 are outside this change.

The accepted workflow has separate files and a backed-up old canonical review. It can be reversed without changing scientific V6 artifacts. Do not replace current human reviews with a historical backup unless the user asks to undo the accepted decision.
