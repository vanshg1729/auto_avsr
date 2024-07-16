python demo.py  data.modality=video \
                pretrained_model_path=./checkpoints/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
                file_path=/ssd_scratch/cvit/vanshg/vansh_phrases/1718881781149_0_66740df5bcb54392537d19a8.mp4

python demo.py  data.modality=video \
                pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth \
                file_path=/ssd_scratch/cvit/vanshg/vansh_phrases/processed_videos/1719055516858_37_6676b49c9cb47e5d1fc7a569.mp4

python demo.py  data.modality=video \
                pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth \
                file_path=/ssd_scratch/cvit/vanshg/lipread_mp4/THIRD/test/THIRD_00043.mp4

python eval.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/preprocessed_grid/video \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/preprocessed_grid/labels/s1_label.csv \
               pretrained_model_path=./checkpoints/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
               verbose=True

python train_grid.py exp_dir=/ssd_scratch/cvit/vanshg/test \
               exp_name=s1_auto_avsr \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/preprocessed_grid/video \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/preprocessed_grid/labels/s1_label.csv \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/preprocessed_grid/labels/s1_label.csv \
               trainer.num_nodes=1 \
               pretrained_model_path=./checkpoints/vsr_trlrwlrs2lrs3vox2avsp_base.pth \

python train_phrase.py exp_dir=/ssd_scratch/cvit/vanshg/vansh_phrases_exp \
               exp_name=vansh_phrases_auto_avsr \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/phrases_dataset \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/phrases_dataset/akshat_phrases/train_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/phrases_dataset/akshat_phrases/test_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune.py exp_dir=/ssd_scratch/cvit/vanshg/vansh_phrases_exp \
               exp_name=vansh_phrases_auto_avsr \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/vansh_phrases \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/vansh_phrases/train_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/vansh_phrases/test_labels.txt \
               trainer.num_nodes=1 \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth \
