

# SUBSET Dataset; one initialization per camera type (center, left, right) with STACK from PREDICTED camera type annotations; predicted annotations
python optimize.py --hparams configs/challenge_test/test_left_pred_optimal_extrem_pred.json --output_dir experiments --visualize_results && python -m evaluation.summarize_batch_results --dir_results experiments/test_left_pred_optimal_extrem_pred
python optimize.py --hparams configs/challenge_test/test_right_pred_optimal_extrem_pred.json --output_dir experiments --visualize_results && python -m evaluation.summarize_batch_results --dir_results experiments/test_right_pred_optimal_extrem_pred
python optimize.py --hparams configs/challenge_test/test_center_pred_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/test_center_pred_optimal_extrem_pred
python -m evaluation.stack_from_individual_results --result_dir_base experiments/ --subset_glob "test_?_pred_optimal_extrem_pred" --subsets center left right
python -m evaluation.summarize_batch_results --from_json --dir_results experiments/test_stacked_pred_optimal_extrem_pred
python -m evaluation.evaluate --dir_results experiments/test_stacked_pred_optimal_extrem_pred --dataset_dir data/extremities/gt/test --taus 0.025 0.02 0.018 0.017 0.016 0.015 0.013

# FULL Dataset; multiple initializations (center, left, right) with ARGMIN; predicted annotations
python optimize.py --hparams configs/challenge_test/test_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/center --overwrite_init_cam_distr "Main camera center" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/center/test_full_optimal_extrem_pred
python optimize.py --hparams configs/challenge_test/test_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/left --overwrite_init_cam_distr "Main camera left" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/left/test_full_optimal_extrem_pred
python optimize.py --hparams configs/challenge_test/test_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/right --overwrite_init_cam_distr "Main camera right" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/right/test_full_optimal_extrem_pred
python -m evaluation.argmin_from_individual_results --result_dir_base experiments/overwrite_init_cam_distr --subset_glob "?/test_full_optimal_extrem_pred" --subsets center left right
python -m evaluation.summarize_batch_results --from_json --dir_results experiments/overwrite_init_cam_distr/argmin/test_full_optimal_extrem_pred/
python -m evaluation.evaluate --dataset_dir data/extremities/gt/test --dir_results experiments/overwrite_init_cam_distr/argmin/test_full_optimal_extrem_pred --taus 0.025 0.02 0.018 0.017 0.016 0.015 0.013

python -m evaluation.write_output_evalai_server --tau 0.016 --per_sample_output experiments/overwrite_init_cam_distr/argmin/test_full_optimal_extrem_pred/per_sample_output.json

# FULL Dataset; multiple initializations (center, left, right) with ARGMIN; gt annotations
python optimize.py --hparams configs/challenge_test/test_full_optimal_extrem_gt.json --output_dir experiments/overwrite_init_cam_distr/center --overwrite_init_cam_distr "Main camera center" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/center/test_full_optimal_extrem_gt
python optimize.py --hparams configs/challenge_test/test_full_optimal_extrem_gt.json --output_dir experiments/overwrite_init_cam_distr/left --overwrite_init_cam_distr "Main camera left" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/left/test_full_optimal_extrem_gt
python optimize.py --hparams configs/challenge_test/test_full_optimal_extrem_gt.json --output_dir experiments/overwrite_init_cam_distr/right --overwrite_init_cam_distr "Main camera right" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/right/test_full_optimal_extrem_gt
python -m evaluation.argmin_from_individual_results --result_dir_base experiments/overwrite_init_cam_distr --subset_glob "?/test_full_optimal_extrem_gt" --subsets center left right
python -m evaluation.summarize_batch_results --from_json --dir_results experiments/overwrite_init_cam_distr/argmin/test_full_optimal_extrem_gt
python -m evaluation.evaluate --dataset_dir data/extremities/gt/test --dir_results experiments/overwrite_init_cam_distr/argmin/test_full_optimal_extrem_gt --taus 0.025 0.02 0.018 0.017 0.016 0.015 0.013

