% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Extended Figure 4 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
load('rad_1979_2022.mat');load('tas_1979_2022.mat');load('color_vartime.mat');load('mask_CONUS_0125d.mat');load('color_blue2red.mat');load('color_relation_SMM_SM.mat')

acf_rad = nan(200,464,21);
for ir = 1:200
    for ic = 1:464
        ts = squeeze(rad(ir,ic,:,:));
        if ~all(isnan(ts(:)))
        [a,b]=autocorr(ts(:));
        acf_rad(ir,ic,:)=a;
        end
    end
end

acf_tas = nan(200,464,21);
for ir = 1:200
    for ic = 1:464
        ts = squeeze(tas(ir,ic,:,:));
        if ~all(isnan(ts(:)))
        [a,b]=autocorr(ts(:));
        acf_tas(ir,ic,:)=a;
        end
    end
end

thr=1;
[ts_rad,Xpatch_rad,Ypatch_rad]=get_uncertainty_fordraw(acf_rad(:,:,2:end),thr,mask_CONUS_0125d);
[ts_tas,Xpatch_tas,Ypatch_tas]=get_uncertainty_fordraw(acf_tas(:,:,2:end),thr,mask_CONUS_0125d);

%% plot figure
filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);x = 1:30;x_patch = cat(2,x,fliplr(x));

load('color_vartime14.mat');load('color_blue2red.mat');fsize = 17; marksize = 15; linewid = 2;
fig = figure('color','white');set(gcf,'Units','centimeters','Position',[5 13 20 14]);
hori = 0.7; wid = 0.7;
pos_a=[0.15 0.15 wid hori];

ax1=subplot('Position',pos_a);
patch(ax1,Xpatch_tas,Ypatch_tas,[0.86,0.05,0.20],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h1=plot(ax1,ts_tas,'.-','linewidth',linewid,'markersize',marksize,'color',[0.86,0.05,0.20]);hold on;

patch(ax1,Xpatch_rad,Ypatch_rad,[0.20,0.47,0.81],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h2=plot(ax1,ts_rad,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.20,0.47,0.81]);hold on;

xlim([0,20]);ylim([0.4,1])
xlabel('Lag','FontSize',fsize,'FontName','Arial')
ylabel('Autocorrelation function','FontSize',fsize,'FontName','Arial')

set(gca,'xtick',1:3:20,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
legend([h1,h2],{'Temperature','Radiation'},'NumColumns',2,'fontsize',fsize)

function [output_mm,x_patch,y_patch]=get_uncertainty_fordraw(input,thr,mask)
leadtimes = size(input,3); output_mm = nan(leadtimes,1);output_std = nan(leadtimes,1);
for ilead = 1:leadtimes
    temp = squeeze(input(:,:,ilead));
    temp(~mask)=nan;
    output_mm(ilead)=nanmean(temp,[1,2]);
    output_std(ilead)=nanstd(temp,[],[1,2]);
end
x1 = 1:leadtimes;x_patch = cat(2,x1,fliplr(x1));
uper = output_mm+thr*output_std;    down = output_mm-thr*output_std;    y_patch = cat(1,uper,flipud(down));
end