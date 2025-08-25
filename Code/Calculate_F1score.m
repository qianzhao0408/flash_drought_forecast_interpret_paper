% ** README **
% Qian Zhao, 08/24/2025
% This script is used to calculate the F1score of the paper 'Deep learning for flash drought prediction and interpretation'
% The outputs of IT-Drought were available by contacting the corresponding author
% The SMroot data was not provided due to lack of license, but you can directly download from the websites
% The auxiliary data of masks were not provided here
% Welcome to cite our paper and Zenodo
% Please contact the corresponding author if any questions
%%
load('SMroot_1979_2022.mat'); load('FlashDrought_NLDAS_testperiod.mat');load('mask_CONUS_0125d.mat');rootDL = '/DL_output/';
aggstep = 7; up_threshold=0.4;   dn_threshold=0.2; days = 365;  yrs = 2022-1980+1;  leadtimes=46;

SM_max = nanmax(SMroot(:));SM_min = nanmin(SMroot(:));
SMobs_pend=aggregate2weekly(SMroot,aggstep);
SMroot_test = SMroot(:,:,:,[2,44]);clear SMroot
[rws,cls,~,~]=size(SMroot_test); aggnum = floor(days/aggstep);

DL_F1score_FD_twotest = nan(rws,cls,leadtimes);DL_recall_FD_twotest = nan(rws,cls,leadtimes);DL_precision_FD_twotest = nan(rws,cls,leadtimes);

for ilead= 1:leadtimes

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
    DLfill1 = SMroot_test(:,:,1:ilead-1,1);    DLfill2 = SMroot_test(:,:,365-leadtimes+ilead+1:end,end);
    DLoutput2 = cat(3,DLfill1,DLoutput_daily,DLfill2);
    DLoutput3 = reshape(DLoutput2,[rws,cls,days,yrs]);

    DLoutput_pend=aggregate2weekly(DLoutput3,aggstep);
    percentile_input = cat(4,SMobs_pend(:,:,:,1),DLoutput_pend);

    percentile_output=get_percentile(percentile_input);
    period_test1 = 19:22; period_test2 = 41:44;

    wks=aggnum*4;
    [DL_NLDASroot_droughtBinary_FD_test1,DL_NLDASroot_drought_FD_test1]=get_binary_dro_r0528(percentile_output(:,:,:,period_test1),dn_threshold,up_threshold,wks);
    [DL_NLDASroot_droughtBinary_FD_test2,DL_NLDASroot_drought_FD_test2]=get_binary_dro_r0528(percentile_output(:,:,:,period_test2),dn_threshold,up_threshold,wks);

    for ir = 1:rws
        for ic = 1:cls
            if mask_CONUS_0125d(ir,ic)

                ts_DL_test1 = squeeze(DL_NLDASroot_droughtBinary_FD_test1(ir,ic,:));
                ts_DL_test2 = squeeze(DL_NLDASroot_droughtBinary_FD_test2(ir,ic,:));
                ts_DL_twotest = cat(1,ts_DL_test1,ts_DL_test2(1:end-6));

                ts_obs_test1 = squeeze(obs_NLDAS_droughtBinary_FD_test1(ir,ic,:));
                ts_obs_test2 = squeeze(obs_NLDAS_droughtBinary_FD_test2(ir,ic,:));
                ts_obs_twotest = cat(1,ts_obs_test1,ts_obs_test2(1:end-6));

                if ~all(isnan(ts_obs_twotest))&&~all(isnan(ts_DL_twotest))
                    TP = find(ts_obs_twotest==1&ts_DL_twotest==1);
                    FP = find(ts_obs_twotest==0&ts_DL_twotest==1);
                    TN = find(ts_obs_twotest==0&ts_DL_twotest==0);
                    FN = find(ts_obs_twotest==1&ts_DL_twotest==0);

                    precision = numel(TP)/(numel(TP)+numel(FP));
                    recall = numel(TP)/(numel(TP)+numel(FN));
                    DL_F1score_FD_twotest(ir,ic,ilead) = 2* (precision * recall) / (precision + recall);
                    DL_recall_FD_twotest(ir,ic,ilead) = recall;
                    DL_precision_FD_twotest(ir,ic,ilead) = precision;

                elseif ~all(isnan(ts_obs_twotest))&&all(isnan(ts_DL_twotest))
                    DL_F1score_FD_twotest(ir,ic,ilead) =0;
                    DL_recall_FD_twotest(ir,ic,ilead) = 0;
                    DL_precision_FD_twotest(ir,ic,ilead) =0;
                end
            end
        end
    end

end


function [output]=aggregate2weekly(input,window)
[rws,cls,dys,yrs]=size(input);
num = floor(dys/window);
output = nan(rws,cls,num,yrs);
a = 1;
for ipentad = 1:window:dys
    if a<=num
        temp = nanmean(input(:,:,ipentad:ipentad+window-1,:),3);
        output(:,:,a,:) = temp;
        a = a+1;
    end
end
end

function [output]=get_percentile(input)
[rws,cls,pends,yrs]=size(input);
output = nan(rws,cls,pends,yrs);
for ir = 1:rws
    for ic = 1:cls
        ts = squeeze(input(ir,ic,:,:));
        if ~all(isnan(ts(:))|ts(:)==0)
            for ipend = 1:pends
                temp = squeeze(ts(ipend,:));
                ts_percentile = cdf('Normal',temp(:),nanmean(temp(:)),nanstd(temp(:)));
                output(ir,ic,ipend,:) =ts_percentile;
            end
        end
    end
end
end

function [DL_NLDASroot_droughtBinary_FD,DL_NLDASroot_drought_FD]=get_binary_dro_r0528(DLoutput_percentile,dn_threshold,up_threshold,wks)
[rws,cls,~,~]=size(DLoutput_percentile);
DL_NLDASroot_droughtBinary_FD  = nan(rws,cls,wks);
DL_NLDASroot_drought_FD= nan(rws,cls,20,6);

for ir = 1:rws
    for ic = 1:cls
        SM_percentile_ts = squeeze(DLoutput_percentile(ir,ic,:,:));

        if ~all(isnan(SM_percentile_ts)|SM_percentile_ts==0)
            site_below20 = find(SM_percentile_ts<dn_threshold);

            if ~isempty(site_below20)
                i = 0; FDdrought=[]; site_end=0; FDdroughtBinary = zeros(wks,1);

                while ~isempty(site_below20)
                    site_20pt = site_below20(1);
                    ts1 = SM_percentile_ts(site_end+1:site_20pt);     site_above40 = find(ts1>up_threshold);
                    ts2 = SM_percentile_ts(site_20pt:end);     site_above20 = find(ts2>dn_threshold);

                    if ~isempty(site_above40)&&~isempty(site_above20)

                        site_start = site_above40(end)+site_end+1;
                        site_end = site_above20(1)+site_20pt-2;
                        drought_length = site_end-site_start+1;

                        onset_speed_a =  (0.4 - ts2(1))/(site_20pt-site_start+1)*100;

                        if onset_speed_a>=5&&drought_length>=4&&drought_length<=12
                            for i_len =1:drought_length
                                speed_temp = (0.4 - ts2(1+i_len))/(site_20pt+i_len-site_start+1)*100;
                                percentile_change = ts2(1+i_len)-ts2(i_len);
                                if speed_temp<5||percentile_change>0
                                    site_onsetend = site_20pt+i_len-1;
                                    break;
                                end
                            end

                            onset_length = site_onsetend-site_start+1;
                            onset_speed = (0.4- SM_percentile_ts(site_onsetend))/onset_length*100;
                            recovery_length = site_end-site_onsetend+1;
                            recovery_speed = (0.2- SM_percentile_ts(site_onsetend))/recovery_length*100;

                            i =i+1;
                            FDdrought(i,1) = site_start;                        FDdrought(i,2) = onset_length;
                            FDdrought(i,3) = onset_speed;                        FDdrought(i,4) = recovery_length;
                            FDdrought(i,5) = recovery_speed;                        FDdrought(i,6) = drought_length;
                            FDdroughtBinary(site_start:site_end)=1;
                        end
                        site_below20(site_below20<=site_end)=[];
                    else
                        site_below20(site_below20==site_20pt)=[];
                    end

                end

                if i>=1
                    DL_NLDASroot_drought_FD(ir,ic,1:i,:)= FDdrought;
                    DL_NLDASroot_droughtBinary_FD(ir,ic,:)=FDdroughtBinary;

                end
            end
        end
    end
end
end