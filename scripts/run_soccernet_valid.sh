
# FULL Dataset, one camera initialization (center); GT annotations vs. predicted
# RUN 1 and 2
python optimize.py --hparams configs/challenge_valid/val_full_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_full_optimal_extrem_pred/
python optimize.py --hparams configs/challenge_valid/val_full_optimal_extrem_gt.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_full_optimal_extrem_gt/
python -m evaluation.evaluate --dataset_dir data/extremities/gt/valid --dir_results experiments/val_full_optimal_extrem_pred/
python -m evaluation.evaluate --dataset_dir data/extremities/gt/valid --dir_results experiments/val_full_optimal_extrem_gt/

# FULL Dataset; multiple initializations (center, left, right) with ARGMIN; predicted annotations
# RUN 3
python optimize.py --hparams configs/challenge_valid/val_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/center --overwrite_init_cam_distr "Main camera center" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/center/val_full_optimal_extrem_pred/
python optimize.py --hparams configs/challenge_valid/val_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/left --overwrite_init_cam_distr "Main camera left" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/left/val_full_optimal_extrem_pred/
python optimize.py --hparams configs/challenge_valid/val_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/right --overwrite_init_cam_distr "Main camera right" &&  python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/right/val_full_optimal_extrem_pred/
python -m evaluation.argmin_from_individual_results --result_dir_base experiments/overwrite_init_cam_distr --subset_glob "?/val_full_optimal_extrem_pred" --subsets center left right
python -m evaluation.summarize_batch_results --from_json --dir_results experiments/overwrite_init_cam_distr/argmin/val_full_optimal_extrem_pred/
python -m evaluation.evaluate --dataset_dir data/extremities/gt/valid --dir_results experiments/overwrite_init_cam_distr/argmin/val_full_optimal_extrem_pred --taus 0.025 0.02 0.018 0.017 0.016 0.015 0.013

# SUBSET Dataset; one initialization per camera type (center, left, right) with STACK from GT camera type annotations; predicted annotations
# RUN 4
python optimize.py --hparams configs/challenge_valid/val_left_gt_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_left_gt_optimal_extrem_pred/
python optimize.py --hparams configs/challenge_valid/val_right_gt_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_right_gt_optimal_extrem_pred/
python optimize.py --hparams configs/challenge_valid/val_center_gt_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_center_gt_optimal_extrem_pred/
# stack individual results, then find a global tau and evaluate
python -m evaluation.stack_from_individual_results --result_dir_base experiments/ --subset_glob "val_?_gt_optimal_extrem_pred" --subsets center left right
python -m evaluation.summarize_batch_results --from_json --dir_results experiments/val_stacked_gt_optimal_extrem_pred
python -m evaluation.evaluate --dataset_dir data/extremities/gt/valid --dir_results experiments/val_stacked_gt_optimal_extrem_pred --taus 0.025 0.02 0.018 0.017 0.016 0.015 0.013

# SUBSET Dataset; one initialization per camera type (center, left, right) with STACK from PREDICTED camera type annotations; predicted annotations
# RUN 5
python optimize.py --hparams configs/challenge_valid/val_left_pred_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_left_pred_optimal_extrem_pred/
python optimize.py --hparams configs/challenge_valid/val_right_pred_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_right_pred_optimal_extrem_pred/
python optimize.py --hparams configs/challenge_valid/val_center_pred_optimal_extrem_pred.json --output_dir experiments && python -m evaluation.summarize_batch_results --dir_results experiments/val_center_pred_optimal_extrem_pred/
# stack individual results, then find a global tau and evaluate
python -m evaluation.stack_from_individual_results --result_dir_base experiments/ --subset_glob "val_?_pred_optimal_extrem_pred" --subsets center left right
python -m evaluation.summarize_batch_results --from_json --dir_results experiments/val_stacked_pred_optimal_extrem_pred/
python -m evaluation.evaluate --dir_results experiments/val_stacked_pred_optimal_extrem_pred --dataset_dir data/extremities/gt/valid --taus 0.025 0.02 0.018 0.017 0.016 0.015 0.013
