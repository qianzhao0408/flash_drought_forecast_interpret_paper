% ** README **
% Qian Zhao, 08/24/2025
% This script is used to calculate the ACC of the paper 'Deep learning for flash drought prediction and interpretation'
% The outputs of IT-Drought were available by contacting the corresponding author
% The SMroot data was not provided due to lack of license, but you can directly download from the websites
% The auxiliary data of masks were not provided here
% Welcome to cite our paper and Zenodo
% Please contact the corresponding author if any questions
%%
load('mask_CONUS_0125d.mat');  rootDL = '/DL_output/';
load('SMroot_1979_2022.mat'); SM_max = nanmax(SMroot(:));SM_min = nanmin(SMroot(:));
obs_anomaly_daily = get_anomaly_forecast(SMroot(:,:,:,2:end)); clear SMroot
obs_anomaly_daily_twotest = obs_anomaly_daily(:,:,:,[18:21,40:43]); clear obs_anomaly_daily
yr_start = 1980; yr_end = 2022; leadtimes = 46;[rws,cls] = size(mask_CONUS_0125d);
DL_ACC_daily_twotest = nan(rws,cls,leadtimes);DL_ACC_daily_test1= nan(rws,cls,leadtimes);DL_ACC_daily_test2= nan(rws,cls,leadtimes);

for ilead = 1:leadtimes
    filenameDL_split1_4 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_1-4_daily_window.npy'];
    DLoutput_split1_4 = readNPY(filenameDL_split1_4); DLoutput_split1_ilead = squeeze(DLoutput_split1_4(:,:,1:365*2,ilead));clear DLoutput_split1_4

    filenameDL_split3_6 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_3-6_daily_window.npy'];
    DLoutput_split3_6 = readNPY(filenameDL_split3_6); DLoutput_split3_ilead = squeeze(DLoutput_split3_6(:,:,1:365*2,ilead));clear DLoutput_split3_6

    filenameDL_split5_8 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_5-8_daily_window.npy'];
    DLoutput_split5_8 = readNPY(filenameDL_split5_8); DLoutput_split5_ilead = squeeze(DLoutput_split5_8(:,:,1:365*2,ilead));clear DLoutput_split5_8

    filenameDL_split7_10 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_7-10_daily_window.npy'];
    DLoutput_split7_10 = readNPY(filenameDL_split7_10); DLoutput_split7_ilead = squeeze(DLoutput_split7_10(:,:,1:365*2,ilead));clear DLoutput_split7_10

    filenameDL_split9_12 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_9-12_daily_window.npy'];
    DLoutput_split9_12 = readNPY(filenameDL_split9_12); DLoutput_split9_ilead = squeeze(DLoutput_split9_12(:,:,1:365*2,ilead));clear DLoutput_split9_12

    filenameDL_split11_14 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_11-14_daily_window.npy'];
    DLoutput_split11_14 = readNPY(filenameDL_split11_14); DLoutput_split11_ilead = squeeze(DLoutput_split11_14(:,:,1:365*2,ilead));clear DLoutput_split11_14

    filenameDL_split13_16 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_13-16_daily_window.npy'];
    DLoutput_split13_16 = readNPY(filenameDL_split13_16); DLoutput_split13_ilead = squeeze(DLoutput_split13_16(:,:,1:365*2,ilead));clear DLoutput_split13_16

    filenameDL_split15_18 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_15-18_daily_window.npy'];
    DLoutput_split15_18 = readNPY(filenameDL_split15_18); DLoutput_split15_ilead = squeeze(DLoutput_split15_18(:,:,1:365*2,ilead));clear DLoutput_split15_18

    filenameDL_split17_20 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_17-20_daily_window.npy'];
    DLoutput_split17_20 = readNPY(filenameDL_split17_20); DLoutput_split17_ilead = squeeze(DLoutput_split17_20(:,:,1:365*2,ilead));clear DLoutput_split17_20

    filenameDL_split19_22 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_19-22_daily_window.npy'];
    DLoutput_split19_22 = readNPY(filenameDL_split19_22); DLoutput_split19_ilead = squeeze(DLoutput_split19_22(:,:,1:365*2,ilead));clear DLoutput_split19_22

    filenameDL_split21_24 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_21-24_daily_window.npy'];
    DLoutput_split21_24 = readNPY(filenameDL_split21_24); DLoutput_split21_ilead = squeeze(DLoutput_split21_24(:,:,1:365*2,ilead));clear DLoutput_split21_24

    filenameDL_split23_26 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_23-26_daily_window.npy'];
    DLoutput_split23_26 = readNPY(filenameDL_split23_26); DLoutput_split23_ilead = squeeze(DLoutput_split23_26(:,:,1:365*2,ilead));clear DLoutput_split23_26

    filenameDL_split25_28 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_25-28_daily_window.npy'];
    DLoutput_split25_28 = readNPY(filenameDL_split25_28); DLoutput_split25_ilead = squeeze(DLoutput_split25_28(:,:,1:365*2,ilead));clear DLoutput_split25_28

    filenameDL_split27_30 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_27-30_daily_window.npy'];
    DLoutput_split27_30 = readNPY(filenameDL_split27_30); DLoutput_split27_ilead = squeeze(DLoutput_split27_30(:,:,1:365*2,ilead));clear DLoutput_split27_30

    filenameDL_split29_32 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_29-32_daily_window.npy'];
    DLoutput_split29_32 = readNPY(filenameDL_split29_32); DLoutput_split29_ilead = squeeze(DLoutput_split29_32(:,:,1:365*2,ilead));clear DLoutput_split29_32

    filenameDL_split31_34 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_31-34_daily_window.npy'];
    DLoutput_split31_34 = readNPY(filenameDL_split31_34); DLoutput_split31_ilead = squeeze(DLoutput_split31_34(:,:,1:365*2,ilead));clear DLoutput_split31_34

    filenameDL_split33_36 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_33-36_daily_window.npy'];
    DLoutput_split33_36 = readNPY(filenameDL_split33_36); DLoutput_split33_ilead = squeeze(DLoutput_split33_36(:,:,1:365*2,ilead));clear DLoutput_split33_36

    filenameDL_split35_38 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_35-38_daily_window.npy'];
    DLoutput_split35_38 = readNPY(filenameDL_split35_38); DLoutput_split35_ilead = squeeze(DLoutput_split35_38(:,:,1:365*2,ilead));clear DLoutput_split35_38

    filenameDL_split37_40 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_37-40_daily_window.npy'];
    DLoutput_split37_40 = readNPY(filenameDL_split37_40); DLoutput_split37_ilead = squeeze(DLoutput_split37_40(:,:,1:365*2,ilead));clear DLoutput_split37_40

    filenameDL_split39_42 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_39-42_daily_window.npy'];
    DLoutput_split39_42 = readNPY(filenameDL_split39_42); DLoutput_split39_ilead = squeeze(DLoutput_split39_42(:,:,1:365*2,ilead));clear DLoutput_split39_42

    filenameDL_split41_44 = [rootDL,'tf_layer4_decoder3_mae_input_365_predict_46_future_0_step_1_0_sample_0.01_random_dim_48_16_lr_1e-4_0125d_dmodel_9_noagg_epoch_61_split_41-44_daily_window.npy'];
    DLoutput_split41_44 = readNPY(filenameDL_split41_44); DLoutput_split41_ilead = squeeze(DLoutput_split41_44(:,:,:,ilead));clear DLoutput_split41_44

    DLoutput_daily = cat(3,DLoutput_split1_ilead,DLoutput_split3_ilead,DLoutput_split5_ilead,DLoutput_split7_ilead,DLoutput_split9_ilead,DLoutput_split11_ilead,DLoutput_split13_ilead,DLoutput_split15_ilead,DLoutput_split17_ilead,DLoutput_split19_ilead, ...
        DLoutput_split21_ilead,DLoutput_split23_ilead,DLoutput_split25_ilead,DLoutput_split27_ilead,DLoutput_split29_ilead,DLoutput_split31_ilead,DLoutput_split33_ilead,DLoutput_split35_ilead,DLoutput_split37_ilead,DLoutput_split39_ilead,DLoutput_split41_ilead);

    DLoutput_daily = DLoutput_daily*(SM_max-SM_min)+SM_min;

    temp1 = cat(3,nan(rws,cls,ilead-1),DLoutput_daily,nan(rws,cls,leadtimes-ilead));
    temp2 = reshape(temp1,rws,cls,365,yr_end-yr_start+1);
    DL_foredata_anomaly_daily = get_anomaly_forecast(temp2);
    DL_foredata_anomaly_daily_twotest = DL_foredata_anomaly_daily(:,:,:,[18:21,40:43]);clear DL_foredata_anomaly_daily

    endweek = 365*8-45+ilead-1;
    endweek2 = 365*4-45+ilead-1;

    ACC_DLagainstNLDAS_daily_twotest = get_acc_forecast_DL(obs_anomaly_daily_twotest,DL_foredata_anomaly_daily_twotest,mask_CONUS_0125d,1:endweek);
    ACC_DLagainstNLDAS_daily_test1 = get_acc_forecast_DL(obs_anomaly_daily_twotest(:,:,:,1:4),DL_foredata_anomaly_daily_twotest(:,:,:,1:4),mask_CONUS_0125d,1:365*4);
    ACC_DLagainstNLDAS_daily_test2= get_acc_forecast_DL(obs_anomaly_daily_twotest(:,:,:,5:8),DL_foredata_anomaly_daily_twotest(:,:,:,5:8),mask_CONUS_0125d,1:endweek2);

    DL_ACC_daily_twotest(:,:,ilead) = ACC_DLagainstNLDAS_daily_twotest(:,:,1);
    DL_ACC_daily_test1(:,:,ilead) = ACC_DLagainstNLDAS_daily_test1(:,:,1);
    DL_ACC_daily_test2(:,:,ilead) = ACC_DLagainstNLDAS_daily_test2(:,:,1);

end
save('ACC_ITDrought.mat','DL_ACC_daily_twotest','DL_ACC_daily_test1','DL_ACC_daily_test2')

function [output]=get_anomaly_forecast(input)
yrs=size(input,4);
input_mm = nanmean(input,4);
input_mm_rep = repmat(input_mm,1,1,1,yrs);
output = input-input_mm_rep;
end

function [output]=get_acc_forecast_DL(input_obs,input_mod,mask,period)
[rws,cls,wks,yrs] = size(input_obs);
input_obs = reshape(input_obs,rws,cls,wks*yrs);
input_mod = reshape(input_mod,rws,cls,wks*yrs);
output = nan(rws,cls,2);
for ir = 1:rws
    for ic = 1:cls
        ts_obs = squeeze(input_obs(ir,ic,period));
        ts_fore = squeeze(input_mod(ir,ic,period));
        if mask(ir,ic)&&nanstd(ts_fore)>0.1
            [acc_dl,p_dl] = corrcoef(ts_obs,ts_fore,'Rows','complete');
            output(ir,ic,1) = acc_dl(2);
            output(ir,ic,2) = p_dl(2);
        end
    end
end
end