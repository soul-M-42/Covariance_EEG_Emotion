run=62
seg_att=15
msFilter_timeLen=3
dilation_array='[1,3,6,12]'
restart_times=30
ext_wd=0.00015
fine_wd=0.00015
# mlp_wds=(0.001 0.0022 0.005 0.011 0.025 0.056 0.125)
mlp_wds=(0.001 0.0022 0.005 0.0075 0.011)
gpus='[0]'

dataset=FACED_blink_only
dataset_Source=FACED_blink_only
dataset_Target=SEEDV
model=cnn_att
# mode='time'
valid_method=1
valid_method_finetune=3
logging=default
iftest=False
proj_name="$dataset_Source""_$dataset_Target""_epoch30""_$model"
# exp_name="segatt$seg_att"
# exp_name="mslen$msFilter_timeLen"
# exp_name="ext_epoch30"
# exp_name="extwd$ext_wd"
exp_name="onlytrain_""$model""_restart$restart_times"
# exp_name="stdmodel_30_15_restart$restart_times"


# 1.train backbone ext on source domain
echo "proj: $proj_name exp:$exp_name run: $run gpus: $gpus valid_method: $valid_method"
echo "train ext wd: $ext_wd"
python train_ext_align.py log.run=$run log.proj_name=$proj_name data=$dataset_Source model=$model\
                    data_source=$dataset_Source\
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name  \
                    model.msFilter_timeLen=$msFilter_timeLen\
                    train.wd=$ext_wd \
                    model.dilation_array=$dilation_array \
                    train.restart_times=$restart_times \
                    model.seg_att=$seg_att\
                    # model.attention_mode=$mode\
                    # -c job
                    # model.timeFilterLen=60 model.msFilter_timeLen=3 \
                    # model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6

# 1.5 compute centroid of covariance matrix of samples from source domain on Riemannian manifold
echo "Computting centroid of source domain"
python compute_centroid.py log.run=$run log.proj_name=$proj_name data=$dataset_Source model=$model\
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name  \
                    model.msFilter_timeLen=$msFilter_timeLen\
                    train.wd=$ext_wd \
                    model.dilation_array=$dilation_array \
                    train.restart_times=$restart_times \
                    model.seg_att=$seg_att\

# 2.finetune preceding transform on target domain
echo "finetune covariance transform with wd: $fine_wd"
python finetune_cov.py log.run=$run log.proj_name=$proj_name data=$dataset_Target model=$model\
                    data_source=$dataset_Source\
                    train.gpus=$gpus train.valid_method=$valid_method_finetune \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name  \
                    model.msFilter_timeLen=$msFilter_timeLen\
                    train.wd=$ext_wd \
                    model.dilation_array=$dilation_array \
                    train.restart_times=$restart_times \
                    model.seg_att=$seg_att\
                    # model.attention_mode=$mode\
                    # -c job
                    # model.timeFilterLen=60 model.msFilter_timeLen=3 \
                    # model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6

# 3.extract feature with concacted model
# echo "extract fea with wd: $ext_wd"
# python extract_fea.py log.run=$run log.proj_name=$proj_name data=$dataset model=$model\
#                       train.gpus=$gpus train.valid_method=$valid_method \
#                       hydra/job_logging=$logging train.iftest=$iftest \
#                       log.exp_name=$exp_name \
#                       model.seg_att=$seg_att\
#                       model.msFilter_timeLen=$msFilter_timeLen\
#                       train.wd=$ext_wd \
#                       model.dilation_array=$dilation_array \
#                       train.restart_times=$restart_times \
#                     #   model.timeFilterLen=60 model.msFilter_timeLen=3 \
#                     #   model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6

# for mlp_wd in "${mlp_wds[@]}"
# do
#   # exp_name="extwd$ext_wd""_mlpwd$mlp_wd"
#   echo "exp_name: $exp_name"
#   echo "Training MLP with mlp_wd: $mlp_wd and ext_wd: $ext_wd"
#   python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset model=$model\
#                     train.gpus=$gpus train.valid_method=$valid_method \
#                     hydra/job_logging=$logging train.iftest=$iftest \
#                     log.exp_name=$exp_name \
#                     model.seg_att=$seg_att\
#                     model.msFilter_timeLen=$msFilter_timeLen\
#                     train.wd=$ext_wd \
#                     mlp.wd=$mlp_wd \
#                     model.dilation_array=$dilation_array \
#                     train.restart_times=$restart_times \
#                     # model.timeFilterLen=60 model.msFilter_timeLen=3 \
#                     # model.seg_att=30 model.avgPoolLen=30 model.timeSmootherLen=6
# done