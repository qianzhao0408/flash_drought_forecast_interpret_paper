% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Figure 2 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
clear;clc;close all
load('mask_CONUS_0125d.mat');load('mask_CONUS_025d.mat');load('mask_forF1socre.mat');load('mask_CONUS_1p5d.mat');load('color_blue2red.mat');thr = 0.5;

% for ACC
load('ACC_ITDrought.mat');load('ACC_CMAagainstNLDAS.mat');load('ACC_CNRMagainstNLDAS.mat');load('ACC_ECMWFagainstNLDAS.mat');load('ACC_GEFSagainstNLDAS.mat');load('ACC_HMCRagainstNLDAS.mat');
[tsacc_DLdaily,Xpatchacc_DLdaily,Ypatchacc_DLdaily]=get_uncertainty_fordraw(DL_ACC_daily_twotest,thr,mask_CONUS_0125d);
[tsacc_CMA,Xpatchacc_CMA,Ypatchacc_CMA]=get_uncertainty_fordraw(squeeze(acc_CMAagainstNLDAS100cm_weekly(:,:,1,:)),thr,mask_CONUS_1p5d);
[tsacc_CNRM,Xpatchacc_CNRM,Ypatchacc_CNRM]=get_uncertainty_fordraw(squeeze(acc_CNRMagainstNLDAS100cm_weekly(:,:,1,:)),thr,mask_CONUS_1p5d);
[tsacc_ECMWF,Xpatchacc_ECMWF,Ypatchacc_ECMWF]=get_uncertainty_fordraw(squeeze(acc_ECMWFagainstNLDAS100cm_weekly(:,:,1,:)),thr,mask_CONUS_1p5d);
[tsacc_GEFS,Xpatchacc_GEFS,Ypatchacc_GEFS]=get_uncertainty_fordraw(squeeze(acc_GEFSagainstNLDAS100cm_weekly(:,:,1,:)),thr,mask_CONUS_1p5d);
[tsacc_HMCR,Xpatchacc_HMCR,Ypatchacc_HMCR]=get_uncertainty_fordraw(squeeze(acc_HMCRagainstNLDAS100cm_weekly(:,:,1,:)),thr,mask_CONUS_1p5d);

%  for F1 score
load('F1_ITDrought.mat');load('F1_CMAagainstNLDAS.mat');load('F1_CNRMagainstNLDAS.mat');load('F1_ECMWFagainstNLDAS.mat');load('F1_GEFSagainstNLDAS.mat');load('F1_HMCRagainstNLDAS.mat')
[tsF1_DLdaily,XpatchF1_DLdaily,YpatchF1_DLdaily]=get_uncertainty_fordraw(DL_F1score_FD_twotest,thr,mask_forF1socre);
[tsF1_CMA,XpatchF1_CMA,YpatchF1_CMA]=get_uncertainty_fordraw(F1_CMAaginstNLDAS_FD,thr,mask_CONUS_1p5d);
[tsF1_CNRM,XpatchF1_CNRM,YpatchF1_CNRM]=get_uncertainty_fordraw(F1_CNRMagainstNLDAS_FD,thr,mask_CONUS_1p5d);
[tsF1_ECMWF,XpatchF1_ECMWF,YpatchF1_ECMWF]=get_uncertainty_fordraw(F1_ECMWFagainstNLDAS_FD,thr,mask_CONUS_1p5d);
[tsF1_GEFS,XpatchF1_GEFS,YpatchF1_GEFS]=get_uncertainty_fordraw(F1_GEFSagainstNLDAS_FD,thr,mask_CONUS_1p5d);
[tsF1_HMCR,XpatchF1_HMCR,YpatchF1_HMCR]=get_uncertainty_fordraw(F1_HMCRagainstNLDAS_FD,thr,mask_CONUS_1p5d);

acc_pattern = squeeze(nanmean(DL_ACC_daily_twotest(:,:,1:14),3));
[matrix_ratio_acc,cmap_ratio_acc] = get_color_acc(acc_pattern,mask_CONUS_0125d,color_blue2red);
F1_pattern = squeeze(nanmean(DL_F1score_FD_twotest(:,:,1:14),3));F1_pattern(~mask_forF1socre)=nan;
[matrix_ratio_F1,cmap_ratio_F1] = get_color_acc(F1_pattern,mask_CONUS_0125d,color_blue2red);

%% plot Fig.2
filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);x = 1:30;x_patch = cat(2,x,fliplr(x));

load('color_vartime14.mat');load('color_blue2red.mat');fsize = 17; marksize = 15; linewid = 1.5;
fig = figure('color','white');set(gcf,'Units','centimeters','Position',[3 3 32 24]);
hori = 0.3; wid = 0.38;
pos_a=[0.09 0.60 wid hori+0.03]; pos_b=[0.57 0.60 wid hori+0.03];
pos_c=[0.09 0.19 wid hori-0.017]; pos_d=[0.57 0.19 wid hori-0.017];pos_label = [-0.1,1.04];

% ------------- panel a :  acc time series ------------- 
ax1=subplot('Position',pos_a);
patch(ax1,Xpatchacc_DLdaily,Ypatchacc_DLdaily,[0.86,0.05,0.20],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h1=plot(ax1,tsacc_DLdaily,'.-','linewidth',linewid,'markersize',marksize,'color',[0.86,0.05,0.20]);hold on;

patch(ax1,Xpatchacc_GEFS,Ypatchacc_GEFS,[0.20,0.47,0.81],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h3=plot(ax1,tsacc_GEFS,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.20,0.47,0.81]);hold on;

patch(ax1,Xpatchacc_CMA,Ypatchacc_CMA,[0.39,0.74,0.83],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h4=plot(ax1,tsacc_CMA,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.39,0.74,0.83]);hold on;

patch(ax1,Xpatchacc_CNRM,Ypatchacc_CNRM,[0.60,0.60,0.92],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h5=plot(ax1,tsacc_CNRM,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.60,0.60,0.92]);hold on;

patch(ax1,Xpatchacc_ECMWF,Ypatchacc_ECMWF,color_vartime(4,:),'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h6=plot(ax1,tsacc_ECMWF,'.-','linewidth',linewid,'markersize',marksize,'Color',color_vartime(4,:));hold on;

patch(ax1,Xpatchacc_HMCR,Ypatchacc_HMCR,[0.61,0.82,0.46],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h7=plot(ax1,tsacc_HMCR,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.61,0.82,0.46]);hold on;

xlim([0,46]);ylim([0,1])
xlabel('Lead time (day)','FontSize',fsize,'FontName','Arial')
ylabel('ACC of SM','FontSize',fsize,'FontName','Arial')

set(gca,'xtick',1:5:46,'ytick',0:0.2:1,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
legend([h1,h3,h4,h5,h6,h7],{'InterDL','GEFS','CMA','CNRM','ECMWF','HMCR'},'NumColumns',2,'fontsize',fsize-3,'edgecolor','none','position',[0.27,0.87,0.2,0.03])
text(ax1,'string',"a",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

% ------------- panel b: F1 time series ------------- 
ax3=subplot('Position',pos_b);
patch(ax3,XpatchF1_DLdaily,YpatchF1_DLdaily,[0.86,0.05,0.20],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h1=plot(ax3,tsF1_DLdaily,'.-','linewidth',linewid,'markersize',marksize,'color',[0.86,0.05,0.20]);hold on;

patch(ax3,XpatchF1_GEFS,YpatchF1_GEFS,[0.20,0.47,0.81],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h3=plot(ax3,tsF1_GEFS,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.20,0.47,0.81]);hold on;

patch(ax3,XpatchF1_CMA,YpatchF1_CMA,[0.39,0.74,0.83],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h4=plot(ax3,tsF1_CMA,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.39,0.74,0.83]);hold on;

patch(ax3,XpatchF1_CNRM,YpatchF1_CNRM,[0.60,0.60,0.92],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h5=plot(ax3,tsF1_CNRM,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.60,0.60,0.92]);hold on;

patch(ax3,XpatchF1_ECMWF,YpatchF1_ECMWF,color_vartime(4,:),'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h6=plot(ax3,tsF1_ECMWF,'.-','linewidth',linewid,'markersize',marksize,'Color',color_vartime(4,:));hold on;

patch(ax3,XpatchF1_HMCR,YpatchF1_HMCR,[0.61,0.82,0.46],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h7=plot(ax3,tsF1_HMCR,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.61,0.82,0.46]);hold on;

xlim([0,46]);ylim([0,1])
xlabel('Lead time (day)','FontSize',fsize,'FontName','Arial')
ylabel('F1 of flash drought','FontSize',fsize,'FontName','Arial')

set(gca,'xtick',1:5:46,'ytick',0:0.2:1,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
legend([h1,h3,h4,h5,h6,h7],{'InterDL','GEFS','CMA','CNRM','ECMWF','HMCR'},'NumColumns',2,'fontsize',fsize-3,'edgecolor','none','position',[0.74,0.87,0.2,0.03])
text(ax3,'string',"b",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

% ------------- panel c: acc spatial pattern ------------- 
ax2=axes('Position',pos_c);
s1 = pcolor(ax2,lon,lat,matrix_ratio_acc);
s1.EdgeColor = 'none'; colormap(ax2,cmap_ratio_acc);hold on
geoshow(ax2,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax2,'string',"c",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.09 0.1 wid 0.015]);
yy=1:10;imagesc(ax,yy)
colormap(ax,color_blue2red(1:10,:));xlabel('Average ACC of SM')
set(gca,'xtick',0.5:2:10.5,'xticklabel',0:0.2:1,'ytick',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ------------- panel d: F1 spatial pattern ------------- 
ax4=axes('Position',pos_d);
s1 = pcolor(ax4,lon,lat,matrix_ratio_F1);
s1.EdgeColor = 'none'; colormap(ax4,cmap_ratio_F1);hold on
geoshow(ax4,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax4,'string',"d",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.57 0.1 wid 0.015]);
yy=1:10;imagesc(ax,yy)
colormap(ax,color_blue2red(1:10,:));xlabel('Average F1 of flash drought')
set(gca,'xtick',0.5:2:10.5,'xticklabel',0:0.2:1,'yticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')



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


function [matrix,cmap]=get_color_acc(input,mask_CONUS_0125d,color_bar)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input<0)=0;input(input>1)=1;
input(0<=input&input<=0.1)=11;
input(0.1<input&input<=0.2)=12;
input(0.2<input&input<=0.3)=13;
input(0.3<input&input<=0.4)=14;
input(0.4<input&input<=0.5)=15;
input(0.5<input&input<=0.6)=16;
input(0.6<=input&input<=0.7)=17;
input(0.7<input&input<=0.8)=18;
input(0.8<input&input<=0.9)=19;
input(0.9<=input&input<=1)=20;

input(site_noveg)=21;
input(~mask_CONUS_0125d)=22;
input = input-10;
[indx,indy]=find(mask_CONUS_0125d==1);
numb=1:length(indx);
matrix=nan(rws,cls);
for i=1:length(indx)
    matrix(indx(i),indy(i))=numb(i);
end
input(~mask_CONUS_0125d)=nan;
dmap = nan(rws,cls,3);
for ir =1:rws
    for ic = 1:cls
        id = input(ir,ic);
        if ~isnan(id)
            dmap(ir,ic,:) = color_bar(id,:);
        end
    end
end
cmap = reshape(dmap,[rws*cls,3]);
site = find(isnan(cmap(:,3)));
cmap(site,:)=[];
end
