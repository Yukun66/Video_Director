export CUDA_VISIBLE_DEVICES=1
#!/bin/bash

# # 执行从1到240的配置文件
# for i in {5..8}
# do
#   python null_text_optimize.py --config p2p_generated_yaml_files/v2_p2p_walk_null_text_optimize_$i.yaml
# done


python run_videodirector.py --config yaml_files/v2_p2p_bear_null_text_optimize_3.yaml

