python demo.py  data.modality=video \
                pretrained_model_path=./checkpoints/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
                file_path=/ssd_scratch/cvit/vanshg/vansh_phrases/1718881781149_0_66740df5bcb54392537d19a8.mp4

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
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/vansh_phrases/preprocessed_phrases \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/vansh_phrases/train_phrases_70.json \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/vansh_phrases/test_phrases_30.json \
               trainer.num_nodes=1 \
               pretrained_model_path=./checkpoints/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
