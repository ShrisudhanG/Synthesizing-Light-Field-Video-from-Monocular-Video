# Synthesizing-Light-Field-Video-from-Monocular-Video

Download weights for Reconstruction Network: https://drive.google.com/file/d/1S8ZPAMski9fEJFAYbuypNZeX4M4lW-1z/view?usp=sharing
Download weights for Refinement Network: https://drive.google.com/file/d/10odwvOXCPS53VLIBSGT7b5rypQ1kXuwx/view?usp=sharing
Download weights for RAFT Optical Flow Network(for temporal loss calculation): https://drive.google.com/file/d/1JhDcpDlKW5F-YFfOAVNZfACjSK9abBo_/view?usp=sharing

To test the Light-Field synthesis network run:
'''
python run_test.py --gpu_1 <gpu_to_run_reconstruction_network> --gpu_2 <gpu_to_run_refinement_network> --lf_path <root_directory_containing_LF_sequences> --disp_path <root_directory_containing_monocular_depth> -d <dataset>
'''

To estimate the temporal loss of Light-Field video predictions run:
'''
python run_temporal_loss.py --gpu_1 <gpu_to_run_reconstruction_network> --gpu_2 <gpu_to_run_refinement_network> --lf_path <root_directory_containing_LF_sequences> --disp_path <root_directory_containing_monocular_depth> -d <dataset>
'''
