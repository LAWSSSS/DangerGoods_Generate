# try

## method

name: pose-scores-three-objects

parameters:

- User-confirmed10/12 afternoon snapshot rollback/current-pose-scores/ec23f39c3851f1dd7f61517020d10c5129701b73; core/config/prompts identical bytes. Recorded models wan2.7-image/doubao-seed-2.1-pro (not qwen);13 original photo metadata notes agree.
- Sealant 密封胶罐.jpg (closed left lid), blue oil 油缸.jpg, hydraulic shock 液压减震器.jpg; fixed44/50 backgrounds;3 rounds each,6 cases18 images; one selected primary reference plus original whole BG each round, no previous image input; no red oil or other shock variant. Reference scope restricts historic class collection without changing its source.
- Seeds20261014/20261024/20261019 respectively, same across BG/rounds;2 pool workers. AST early-stop condition disabled; native scoring/size feedback and best-average selection retained; API responses complete, POSTs not retried automatically; independent audit not fed back.

## result

- All6×3 completed;18 raw outputs from18 Wan requests,60 VLM responses; zero failures or pending. Models verified per request.4 offline checks pass; hashes, source snapshot, reference order, output/background ratios and6 selected-image hashes verified.
- Mean tone scores73.0/74.2/65.3; size58.5/41.7/43.0; overall59.7/52.3/46.8. All-three-above85 counts0/0/0. Independent identity matches3/2/3 of6; combined passes2/0/1. Model observations, not measured physics.
- Selected rounds: sealant44R1/50R2; oil44R3/50R1; shock44R2/50R1. Shock44R2 becomes a rust tube yet is selected by average; grading omits an immutable identity gate.
- Run output/validation/pose_scores_three_objects_20261014_140415;3 comparison boards and2 galleries, complete prompts/model responses/ROIs/native outcome/metadata.18 round images exported to output/images and6 selected raw copies to output/final_images; metadata in output/metadata; prefix retest_ec23f39. No commits/pushes, core unchanged, later repairs remain in stash.

## gap

- Generation/testing scope complete. Quality goal remains inconsistent: identity preserved8/18, independent combined3/18; highest average may choose a structurally wrong object. Further changes need identity preservation ahead of tone/size averaging and evidenced physical placements. No additional image batches launched.
