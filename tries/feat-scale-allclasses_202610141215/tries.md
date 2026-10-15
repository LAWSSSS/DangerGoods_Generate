# try

## method

name: feat-scale-allclasses

parameters:

- branch feat/realistic-object-scale; commit b65cc9cc4ee5e57c05a1d2bd612298951ee2a67c; same code as initial rollback/prev-size-range, reused byte/hash-matched earlier results; initial_manifest and selected_branch evidence retained.
- fixed 44/50 backgrounds: prefix2026_08_06_36_6_jianliao1_50_houjian_18#33_zhongfei3_15_zhongfei2_8#33_jianliao2_8#33_; 15 classes, one primary each (see manifest), two BG, exactly3 rounds; 30 cases,90 complete outputs.
- models wan2.7-image / qwen3.8-flash; n=1; watermark=false; base seed20261014+class index; same class seed across BG/rounds; 2 pool workers. Seed does not guarantee deterministic model output.
- old pipeline input R1=original whole BG, R2/R3=original BG+previous output; object photographs only supplied to independent audit, not to Wan or old evaluation. Original generation/config/prompt files equal selected commit bytes; no pasted pixels or output re-encoding.
- single AST early-exit condition suppressed for3 full rounds; raw old evaluation unchanged and baseline_would_stop recorded; old feedback optimization retained; safe transport omits secret logging and automatic POST retries.
- independent audit each round: primary object photo, original BG, entire output1600-long-side JPEG92, planned-position ROI; fresh one-turn dialogue, max_tokens3200/temperature0.3; identity/placement/support/occlusion/size/evidence/uncertainties; only explicit positives, no uncertainties, confidence>=0.65 pass; not 3D ground truth.
- user pause froze controller/workers: 11 complete cases,37 saved images,38 Wan submissions. Resumed after selecting feat branch; one shock50 R3 proxy connection failed. Original attempt124 retained; one final-round recovery, input and RNG rebuilt offline from original5 Qwen receipts, R1/R2 prompts byte matched, original dialogue image hashes restored, first2 PNG hashes unchanged; no unrelated image batch.
- actual experiment output/validation/feat_realistic_object_scale_20261014 is alias to original data folder; report.json / iteration_analysis.json / requests.json / selected_branch.json / initial_manifest.json / final_validation.json, all raw images, per-round prompts/API/model responses, 15 boards and2 HTML galleries.
- check:4 synthetic offline script tests, AST,90 image hashes and source input hashes, all30×3 records and valid old evaluations, Git diff --check. Reporting-only changes to branch guard/request totals after actual test retained separately; tested script snapshot remains original.

## result

- 30/30 complete,90 complete raw API outputs. Wan91 client attempts=90 returned+1 pause-related proxy failure; Qwen210 returned; no requests in flight; no new image batches after completion. Current branch feat/realistic-object-scale, no core pipeline changes, no commit/push.
- old size acceptable R1/R2/R3=1/5/13 of30; old mean quality0.3376666667/0.4016666667/0.4200000000. R3-R1 mean+0.0823333333;15 improve,8 worsen,7 same. These are model size/quality judgments, not actual cm or identity preservation.
- independent reference matches0/90, mismatches79/90, uncertain11/90; full structure/physics/image check passes0/90. Natural integration R1/R2/R3=6/0/0 of30; stable support1/0/0 of30. Qualifiers are model observations, not physical measurements.
- average R3/R1 Laplacian variance ratio3.0941906068; source/output boards show strong later-round noise/texture. Ratio is not image-quality improvement evidence.
- root issues: Wan never sees original object; mandatory same-as-scrap colour/material/rust conflicts with plastic/paint/chrome identity; feedback focuses size; Gaussian corner placement lacks real support proof; previous-output reference carries earlier shape/texture errors. No reliable original-shape preservation after3 rounds.

## gap

Testing goal completed; appearance and physical-quality goal unmet. Need direct immutable original-item visual conditioning and original BG, separate identity from scale feedback, preserve material/main colour, and verify evidenced supports. Additional iterations alone did not solve these issues; no further batch authorized or launched by this record.

