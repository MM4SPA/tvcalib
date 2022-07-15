
# /!\ Best performing variant only
######### FULL=no camera type filtering, but multiple initializations +  argmin for selection
python optimize.py --hparams configs/challenge_challenge/chal_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/center --overwrite_init_cam_distr "Main camera center" && python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/center/chal_full_optimal_extrem_pred 
python optimize.py --hparams configs/challenge_challenge/chal_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/left --overwrite_init_cam_distr "Main camera left" && python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/left/chal_full_optimal_extrem_pred 
python optimize.py --hparams configs/challenge_challenge/chal_full_optimal_extrem_pred.json --output_dir experiments/overwrite_init_cam_distr/right --overwrite_init_cam_distr "Main camera right" && python -m evaluation.summarize_batch_results --dir_results experiments/overwrite_init_cam_distr/right/chal_full_optimal_extrem_pred 
python -m evaluation.argmin_from_individual_results --result_dir_base experiments/overwrite_init_cam_distr --subset_glob "?/chal_full_optimal_extrem_pred" --subsets center left right
python -m evaluation.summarize_batch_results --from_json --dir_results experiments/overwrite_init_cam_distr/argmin/chal_full_optimal_extrem_pred/ --taus inf 0.025 0.02 0.017 0.015 0.013

# zip for eval.ai server 
# Tau selection: if we only consider the mean NDC preprojection loss for filtering, we can manually select
# However, dataset_distr-loss curve looks equal for all splits (valid, test, challenge)
python -m evaluation.write_output_evalai_server --tau 0.017 --per_sample_output experiments/overwrite_init_cam_distr/argmin/chal_full_optimal_extrem_pred/per_sample_output.json
python -m evaluation.write_output_evalai_server --tau 0.016 --per_sample_output experiments/overwrite_init_cam_distr/argmin/chal_full_optimal_extrem_pred/per_sample_output.json
python -m evaluation.write_output_evalai_server --tau 0.015 --per_sample_output experiments/overwrite_init_cam_distr/argmin/chal_full_optimal_extrem_pred/per_sample_output.json
