python demo.py  data.modality=video \
                pretrained_model_path=./checkpoints/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
                file_path=/ssd_scratch/cvit/vanshg/vansh_phrases/1718881781149_0_66740df5bcb54392537d19a8.mp4

python demo.py  data.modality=video \
                pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth \
                file_path=/ssd_scratch/cvit/vanshg/vansh_phrases/processed_videos/1719055516858_37_6676b49c9cb47e5d1fc7a569.mp4

python eval.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/preprocessed_grid/video \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/preprocessed_grid/labels/s1_label.csv \
               pretrained_model_path=./checkpoints/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
               verbose=True

python train_phrase.py exp_dir=/ssd_scratch/cvit/vanshg/vansh_phrases_exp \
               exp_name=vansh_phrases_auto_avsr \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/phrases_dataset \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/phrases_dataset/akshat_phrases/train_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/phrases_dataset/akshat_phrases/test_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/phrases_dataset/akshat_phrases/val_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

# For Deaf Dataset
python finetune_deaf.py exp_dir=/ssd_scratch/cvit/vanshg/deaf_youtube_exp \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/ \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/realdeafdreamer/train_reduced_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/realdeafdreamer/test_reduced_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/realdeafdreamer/val_reduced_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune_deaf.py exp_dir=/ssd_scratch/cvit/vanshg/deaf_youtube_exp \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/ \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/mia_sandra/train_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/mia_sandra/test_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/mia_sandra/val_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/ \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/deafdaydreamer/all_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

# For Accented Dataset
python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/accented_speakers \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/daniel_howell/train_reduced_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune_deaf.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/accented_speakers \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/jack/train_reduced_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/jack/val_reduced_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/jack/test_reduced_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune_deaf.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/accented_speakers \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/diane_jennings/train_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/diane_jennings/val_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/accented_speakers/diane_jennings/test_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune_lip2wav.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chem/train_orig_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chem/test_orig_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chem/val_orig_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

# For Lip2Wav Dataset
python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/lip2wav \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/lip2wav/chem/val_reduced600_10_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune_deaf.py exp_dir=/ssd_scratch/cvit/vanshg/deaf_youtube_exp \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/lip2wav/ \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/datasets/lip2wav/chem/train_reduced3000_50_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/lip2wav/chem/test_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/datasets/lip2wav/chem/val_reduced600_10_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

# For Mead Dataset
python finetune_deaf.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset \
               data.dataset.train_file=/ssd_scratch/cvit/akshat/phrases_mead/w011/front_labels_train_random_80_e.txt \
               data.dataset.test_file=/ssd_scratch/cvit/akshat/phrases_mead/w011/front_labels_test_random_80_e.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/ \
               data.dataset.test_file=/ssd_scratch/cvit/akshat/phrases_mead/w011/front_labels_test_random_80_e.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

# For Lip2Wav Dataset
python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/lip2wav \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/lip2wav/chess/val_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/ \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/benny/train_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth

python finetune.py exp_dir=/ssd_scratch/cvit/vanshg/vansh_phrases_exp \
               exp_name=vansh_phrases_auto_avsr \
               data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/vansh_phrases \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/vansh_phrases/train_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/vansh_phrases/test_labels.txt \
               trainer.num_nodes=1 \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth \