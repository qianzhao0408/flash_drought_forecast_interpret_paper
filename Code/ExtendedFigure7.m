% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Extended Figure 7 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
load('SMroot_mm.mat');load('ACC_ITDrought.mat');load('SlopeSpectrum.mat');load('SoilMoistureMemory.mat');load('pet_value_drought.mat');load('mask_CONUS_0125d.mat');load('color_relation_SMM_SM.mat')

ACC_pattern = squeeze(nanmean(DL_ACC_daily_twotest(:,:,1:14),3));
ACC_points = ACC_pattern(:);

slope_spectrum_points = slope_spectrum_log10(:);
SMM_points = SMM_pattern(:);

site = find(isnan(SMM_points)|isnan(ACC_points)|isnan(slope_spectrum_points));
SMM_points(site)=[];ACC_points(site)=[];slope_spectrum_points(site)=[];
X = cat(2,SMM_points,slope_spectrum_points);

stats_ACC= regstats(ACC_points,X,'linear');

inter = stats_ACC.tstat.beta(1);a = stats_ACC.tstat.beta(2);b = stats_ACC.tstat.beta(3);
pValueSMM =stats_ACC.tstat.pval(2);pValueSpectrum =stats_ACC.tstat.pval(3);
Y_pred = stats_ACC.yhat;

data_ACC = [ACC_points,Y_pred];  density_ACC = ksdensity([data_ACC(:,1),data_ACC(:,2)],[data_ACC(:,1),data_ACC(:,2)]);

SM_value = SMroot_mm(:)/1000;% turn from kg m^{-2} to m^3 m^{-3}
SMM_value = SMM_pattern(:);
PET_value = pet_onset_pattern(:);
ET_value = PET_value.*SM_value;

site_nan = find(isnan(ET_value)|isnan(SMM_value));ET_value(site_nan)=[];SMM_value(site_nan)=[];
figure,plot(ET_value,SMM_value,'.');

stats_SMM= regstats(SMM_value,ET_value,'linear');pValue=stats_SMM.tstat.pval(1);
x_panelc = 0:0.1:2.5;Y_line = stats_SMM.beta(2)*x_panelc+stats_SMM.beta(1);

data_SMM = [ET_value,SMM_value];
density_SMM = ksdensity([data_SMM(:,1),data_SMM(:,2)],[data_SMM(:,1),data_SMM(:,2)]);

%% plot
fig = figure('color','white');set(gcf,'Units','centimeters','Position',[5 5 30 12]); fsize = 17;
hori = 0.7; wid = 0.35; 
pos_a=[0.09 0.2 wid hori];pos_b=[0.59 0.2 wid hori];pos_panel = [-0.13,1.07];pos_label = [-0.13,1.04];

% ------------------ panel a ------------------ 
ax1=axes('Position',pos_a);
scatter(ax1,data_ACC(:,1),data_ACC(:,2),5,density_ACC,'filled');colormap(ax1,flipud(map_listCopy)/255);hold on;
plot(ax1,ACC_points, ACC_points,'-','linewidth',2,'color',[0.85,0.33,0.10]); 
xlabel('True ACC');ylabel('Predicted ACC');
text(0.48,0.44,['ACC=',num2str(a,'%.3f'),'\timest_{SMM}',num2str(b,'%.3f'),'\timesSpectrum+',num2str(inter,'%.3f')],'fontsize',fsize-2)
text(0.66,0.57,'p','fontsize',fsize-2,'fontangle','italic')
text(0.68,0.57,' t_{SMM}<0.001 ','fontsize',fsize-2)
text(0.58,0.51,'p','fontsize',fsize-2,'fontangle','italic')
text(0.60,0.51,[' Spectrum<0.001; R^2=',num2str(stats_ACC.rsquare,'%.2f')],'fontsize',fsize-2)
xlim([0.4,1]);ylim([0.4,1])

set(gca,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
text(ax1,'string',"a",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

% ------------------ panel b ------------------ 
ax2=axes('Position',pos_b);
h3 = scatter(data_SMM(:,1),data_SMM(:,2),5,density_SMM,'filled');hold on
colormap(ax2,flipud(map_listCopy)/255);
plot(ax2,x_panelc,Y_line,'-','linewidth',2,'color',[0.85,0.33,0.10]);
xlabel('SM (m^3 m^{-3}) \times PET (mm)');ylabel('t_{SMM} (day)')
set(gca,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
text(ax2,'string',"b",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');
text(1.5,8,'p','fontsize',fsize,'fontangle','italic')
text(1.58,8,'<0.001','fontsize',fsize)
xlim([0,2.5]);ylim([6,20]);