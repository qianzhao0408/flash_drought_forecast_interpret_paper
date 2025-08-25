% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Figure 3 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
load('IG_precipitation.mat');load('IG_SMsurf.mat');load('IG_radiation.mat');load('IG_SMroot.mat');load('IG_temperature.mat');load('IG_windspeed.mat');load('IG_snow.mat');load('IG_surfacepressure.mat');load('IG_VPD.mat')
load('color_vartime.mat');load('mask_CONUS_0125d.mat');load('color_blue2red_2.mat');load('color_relation_SMM_SM.mat');load('SMroot_mm.mat')

[ts_pr,ts_pr_std]=get_ig_ts_std(ig_onset_FD_pattern_Pr(:,:,:,1));
[ts_rad,ts_rad_std]=get_ig_ts_std(ig_onset_FD_pattern_rad(:,:,:,1));
[ts_SMroot,ts_SMroot_std]=get_ig_ts_std(ig_onset_FD_pattern_SMroot(:,:,:,1));
[ts_SMsurf,ts_SMsurf_std]=get_ig_ts_std(ig_onset_FD_pattern_SMsurf(:,:,:,1));
[ts_snow,ts_snow_std]=get_ig_ts_std(ig_onset_FD_pattern_snow(:,:,:,1));
[ts_sp,ts_sp_std]=get_ig_ts_std(ig_onset_FD_pattern_sp(:,:,:,1));
[ts_tas,ts_tas_std]=get_ig_ts_std(ig_onset_FD_pattern_tas(:,:,:,1));
[ts_vpd,ts_vpd_std]=get_ig_ts_std(ig_onset_FD_pattern_vpd(:,:,:,1));
[ts_vs,ts_vs_std]=get_ig_ts_std(ig_onset_FD_pattern_vs(:,:,:,1));

ts_ig_raw = cat(2,ts_pr,ts_rad,ts_SMsurf,ts_snow,ts_sp,ts_tas,ts_vpd,ts_vs,ts_SMroot);
ts_vars_std = cat(2,ts_pr_std,ts_rad_std,ts_SMsurf_std,ts_snow_std,ts_sp_std,ts_tas_std,ts_vpd_std,ts_vs_std,ts_SMroot_std);

threshold =0.002;SMM_pattern=get_SMM_pattern(ig_onset_FD_pattern_SMroot(:,:,:,1),threshold);
[matrix_ratio_SMM,cmap_ratio_SMM] = get_color_SMM(SMM_pattern,mask_CONUS_0125d,color_blue2red([1:2,4:5,6:7,9:10,11,12],:));

SM_value = SMroot_mm(:)/1000;% turn from kg m^{-2} to m^3 m^{-3}
SMM_value = SMM_pattern(:);
site_nan = find(isnan(SM_value)|isnan(SMM_value));SM_value(site_nan)=[];SMM_value(site_nan)=[];
stats= regstats(SMM_value,SM_value,'linear');pValue=stats.tstat.pval(1);
x_panelc = 0.04:0.02:0.4;Y_line = stats.beta(2)*x_panelc+stats.beta(1);
data = [SM_value,SMM_value];density_2D = ksdensity([data(:,1),data(:,2)],[data(:,1),data(:,2)]);

%% plot Fig.3
filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);
x = 1:30;x_patch = cat(2,x,fliplr(x));

fig = figure('color','white');fsize = 18;set(gcf,'Units','centimeters','Position',[3 3 32 22])
pos1_pre=[0.1 0.13 0.4 0.72];      pos1_SM=[0.08 0.7 0.4 0.45];    pos2=[0.58 0.57 0.34 0.31];pos_label=[0.93 0.57 0.015 0.31];    pos3 = [0.58 0.13 0.36 0.32];       pos_panel = [-0.13,1.07];
mksize = 7;color_smroot = [0.73,0.15,0.19]; color_smsurf = [120,171,48];color_pr = [0.30,0.75,0.93];       color_rad = [1.00,0.41,0.16];color_snow=[0.65,0.65,0.65];       
color_sp=[0.95,0.31,0.71];color_tas = [0.93,0.69,0.13];       color_vpd = [0.97,0.95,0.65];color_vs=[0.82,0.97,0.65];

% *********** panel a: ig decay curve of SM ***********
ax1_1=subplot('Position',pos1_pre);
uper = ts_ig_raw(end-29:end,end)+ts_vars_std(end-29:end,end);    down = ts_ig_raw(end-29:end,end)-ts_vars_std(end-29:end,end);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_blue2red(10,:),'FaceAlpha',.1,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,3)+ts_vars_std(end-29:end,3);    down = ts_ig_raw(end-29:end,3)-ts_vars_std(end-29:end,3);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_blue2red(10,:),'FaceAlpha',.1,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,1)+ts_vars_std(end-29:end,1);    down = ts_ig_raw(end-29:end,1)-ts_vars_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_pr,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,2)+ts_vars_std(end-29:end,2);    down = ts_ig_raw(end-29:end,2)-ts_vars_std(end-29:end,2);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_rad,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,4)+ts_vars_std(end-29:end,4);    down = ts_ig_raw(end-29:end,4)-ts_vars_std(end-29:end,4);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_snow,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,5)+ts_vars_std(end-29:end,5);    down = ts_ig_raw(end-29:end,5)-ts_vars_std(end-29:end,5);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_sp,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,6)+ts_vars_std(end-29:end,6);    down = ts_ig_raw(end-29:end,6)-ts_vars_std(end-29:end,6);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_tas,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,7)+ts_vars_std(end-29:end,7);    down = ts_ig_raw(end-29:end,7)-ts_vars_std(end-29:end,7);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_vpd,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_ig_raw(end-29:end,8)+ts_vars_std(end-29:end,8);    down = ts_ig_raw(end-29:end,8)-ts_vars_std(end-29:end,8);    y_patch = cat(1,uper,flipud(down));
patch(ax1_1,x_patch,y_patch,color_vs,'FaceAlpha',.2,'EdgeAlpha',.0);hold on

plot(ax1_1,[1,30],[0,0],'--k','linewidth',2);hold on
h_SMroot=plot(ax1_1,ts_ig_raw(end-29:end,end),'-','linewidth',2,'color',color_smroot,'MarkerFaceColor',color_blue2red(10,:),'MarkerSize',mksize-2);
h_SMsurf=plot(ax1_1,ts_ig_raw(end-29:end,3),'--','linewidth',2,'color',color_smroot,'Color',color_blue2red(10,:),'MarkerSize',mksize);
h_snow=plot(ax1_1,ts_ig_raw(end-29:end,4),'o-k','linewidth',2,'MarkerFaceColor',color_snow,'MarkerSize',mksize);hold on
h_sp=plot(ax1_1,ts_ig_raw(end-29:end,5),'o-k','linewidth',2,'MarkerFaceColor',color_sp,'MarkerSize',mksize);hold on
h_vs=plot(ax1_1,ts_ig_raw(end-29:end,8),'o-k','linewidth',2,'MarkerFaceColor',color_vs,'MarkerSize',mksize);hold on
h_vpd=plot(ax1_1,ts_ig_raw(end-29:end,7),'o-k','linewidth',2,'MarkerFaceColor',color_vpd,'MarkerSize',mksize);hold on
h_pr=plot(ax1_1,ts_ig_raw(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_pr,'MarkerSize',mksize);hold on
h_tas=plot(ax1_1,ts_ig_raw(end-29:end,6),'o-k','linewidth',2,'MarkerFaceColor',color_tas,'MarkerSize',mksize);hold on
h_rad=plot(ax1_1,ts_ig_raw(end-29:end,2),'o-k','linewidth',2,'MarkerFaceColor',color_rad,'MarkerSize',mksize);hold on

legend([h_SMroot,h_SMsurf,h_pr,h_rad,h_snow,h_sp,h_tas,h_vpd,h_vs], ...
    {'SMroot';'SMsurface';'precipitation';'radiation';'snow';'surface pressure';'temperature';'VPD';'wind speed'},'location','northwest','fontsize',fsize);
xlabel('Time lag (day)');ylabel('IG of feature')
set(gca,'color','none','ycolor',[0,0,0],'xtick',3:3:30,'xticklabel',28:-3:1,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
xlim([0,31]);ylim([-0.012,0.07]);yyaxis right; set(gca,'ycolor','none')
text(ax1_1,'string',"a",'Units','normalized','position',pos_panel,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

% % *********** panel b: tSMM *********** 
ax2=axes('Position',pos2);
s1 = pcolor(ax2,lon,lat,matrix_ratio_SMM);
s1.EdgeColor = 'none'; colormap(ax2,cmap_ratio_SMM);hold on
geoshow(ax2,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax2,'string',"b",'Units','normalized','position',pos_panel,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax1=axes('position',pos_label);
yy=1:8; yy = yy';imagesc(ax1,flipud(yy))
colormap(ax1,color_blue2red([1:2,4:5,6:7,9:10],:));xlabel('t_{SMM} (day)')
set(gca,'ytick',1.5:7.5,'yticklabel',18:-1:12,'yaxislocation','right','xticklabel',[],'FontSize',fsize,'FontName','Arial')

% ****** panel c: relationship between tSMM and SM ****** 
ax3 = axes('Position',pos3);
h3 = scatter(data(:,1),data(:,2),5,density_2D,'filled');hold on
colormap(ax3,flipud(map_listCopy)/255);
plot(ax3,x_panelc,Y_line,'-','linewidth',2,'color',[0.85,0.33,0.10]);
xlabel('SM (m^3 m^{-3})');ylabel('t_{SMM} (day)')
set(gca,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
text(ax3,'string',"c",'Units','normalized','position',pos_panel,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');
text(0.25,14,'p','fontsize',fsize,'fontangle','italic')
text(0.263,14,'<0.001','fontsize',fsize)
xlim([0.04,0.4]);ylim([10,20]);


function [ts,ts_std]=get_ig_ts_std(input)
ts = squeeze(nanmean(input,[1,2]));ts_std = squeeze(nanstd(input,[],[1,2]));
end

function [output]=get_SMM_pattern(input,threshold)
[rws,cls,~] = size(input);
output = nan(rws,cls);
for ir = 1:rws
    for ic = 1:cls
         ts = flipud(squeeze(input(ir,ic,:)));
        if ~all(isnan(ts))
            site = find(ts<=threshold);
            output(ir,ic) = site(1);
        end
    end
end
end

function [matrix,cmap]=get_color_SMM(input,mask_CONUS_0125d,color_ig)

[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(0<=input&input<=12)=1;
input(12<input&input<=13)=2;
input(13<input&input<=14)=3;
input(14<input&input<=15)=4;
input(15<input&input<=16)=5;
input(16<input&input<=17)=6;
input(17<input&input<=18)=7;
input(18<input&input<=20)=8;
input(site_noveg)=9;
input(~mask_CONUS_0125d)=10;

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
            dmap(ir,ic,:) = color_ig(id,:);
        end
    end
end
cmap = reshape(dmap,[rws*cls,3]);
site = find(isnan(cmap(:,3)));
cmap(site,:)=[];
end