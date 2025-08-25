% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Figure 4 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
load('IG_precipitation.mat');load('IG_radiation.mat');load('IG_temperature.mat');load('IG_temperature_season.mat');load('IG_radiation_season.mat');load('IG_precipitation_season.mat')
load('color_vartime.mat');load('mask_CONUS_0125d.mat');load('color_blue2red_2.mat');load('color_relation_SMM_SM.mat');load('color_ig_0112Copy_pr.mat');load('color_ig_0112Copy.mat')

pattern_pr = nanmean(ig_onset_FD_pattern_Pr(:,:,end-6:end,1),3);pattern_rad = nanmean(ig_onset_FD_pattern_rad(:,:,end-6:end,1),3);pattern_tas = nanmean(ig_onset_FD_pattern_tas(:,:,end-6:end,1),3);

[ts_pr_spring,ts_pr_spring_std]=get_ig_ts_std(ig_onset_FD_spring_pattern_pr(:,:,:,1));
[ts_pr_summer,ts_pr_summer_std]=get_ig_ts_std(ig_onset_FD_summer_pattern_pr(:,:,:,1));
[ts_pr_autumn,ts_pr_autumn_std]=get_ig_ts_std(ig_onset_FD_autumn_pattern_pr(:,:,:,1));
[ts_pr_winter,ts_pr_winter_std]=get_ig_ts_std(ig_onset_FD_winter_pattern_pr(:,:,:,1));

[ts_rad_spring,ts_rad_spring_std]=get_ig_ts_std(ig_onset_FD_spring_pattern_rad(:,:,:,1));
[ts_rad_summer,ts_rad_summer_std]=get_ig_ts_std(ig_onset_FD_summer_pattern_rad(:,:,:,1));
[ts_rad_autumn,ts_rad_autumn_std]=get_ig_ts_std(ig_onset_FD_autumn_pattern_rad(:,:,:,1));
[ts_rad_winter,ts_rad_winter_std]=get_ig_ts_std(ig_onset_FD_winter_pattern_rad(:,:,:,1));

[ts_tas_spring,ts_tas_spring_std]=get_ig_ts_std(ig_onset_FD_spring_pattern_tas(:,:,:,1));
[ts_tas_summer,ts_tas_summer_std]=get_ig_ts_std(ig_onset_FD_summer_pattern_tas(:,:,:,1));
[ts_tas_autumn,ts_tas_autumn_std]=get_ig_ts_std(ig_onset_FD_autumn_pattern_tas(:,:,:,1));
[ts_tas_winter,ts_tas_winter_std]=get_ig_ts_std(ig_onset_FD_winter_pattern_tas(:,:,:,1));

[matrix_ratio_vpd,cmap_ratio_vpd] = get_color_tas(pattern_tas*1000,mask_CONUS_0125d,color_igCopy([1:9,11,12],:));
[matrix_ratio_pet,cmap_ratio_pet] = get_color_rad(pattern_rad*1000,mask_CONUS_0125d,color_igCopy([1:9,11,12],:));
[matrix_ratio_pr,cmap_ratio_pr] = get_color_pr(pattern_pr*1000,mask_CONUS_0125d,color_ig_pr);

%% plot Fig. 4
filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);x = 1:30;x_patch = cat(2,x,fliplr(x));

fig = figure('color','white');fsize = 17;
set(gcf,'Units','centimeters','Position',[3 3 32 32])
color_spring = [0.47,0.67,0.19];color_summer = [0.93,0.69,0.13];color_autumn = [255,107,42]/255;color_winter = [77,191,237]/255;

pos1=[0.08 0.74 0.36 0.22];pos2=[0.53 0.74 0.34 0.22];pos3=[0.08 0.42 0.36 0.22];pos4=[0.53 0.42 0.34 0.22];
pos5=[0.08 0.1 0.36 0.22];pos6=[0.53 0.1 0.34 0.22];pos_label = [-0.09,1.1]; pos_varname = [0.45,1.06];

% **************  radiation ************** 
ax1=subplot('Position',pos1);
yyaxis left
plot(ax1,[1,30],[0,0],'--k','linewidth',2);hold on
uper = ts_rad_spring(end-29:end,1)+ts_rad_spring_std(end-29:end,1);    down = ts_rad_spring(end-29:end,1)-ts_rad_spring_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax1,x_patch,y_patch,color_spring,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_rad_summer(end-29:end,1)+ts_rad_summer_std(end-29:end,1);    down = ts_rad_summer(end-29:end,1)-ts_rad_summer_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax1,x_patch,y_patch,color_summer,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_rad_autumn(end-29:end,1)+ts_rad_autumn_std(end-29:end,1);    down = ts_rad_autumn(end-29:end,1)-ts_rad_autumn_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax1,x_patch,y_patch,color_autumn,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_rad_winter(end-29:end,1)+ts_rad_winter_std(end-29:end,1);    down = ts_rad_winter(end-29:end,1)-ts_rad_winter_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax1,x_patch,y_patch,color_winter,'FaceAlpha',.2,'EdgeAlpha',.0);hold on

h1=plot(ax1,ts_rad_spring(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_spring,'MarkerSize',8);hold on
h2=plot(ax1,ts_rad_summer(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_summer,'MarkerSize',8);hold on
h3=plot(ax1,ts_rad_autumn(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_autumn,'MarkerSize',8);hold on
h4=plot(ax1,ts_rad_winter(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_winter,'MarkerSize',8);hold on
legend([h1,h2,h3,h4],{'Spring';'Summer';'Autumn';'Winter'},'location','southwest')
ylabel('IG of radiation (\times10^{-3})')
set(gca,'color','none','ycolor',[0,0,0],'xtick',[1,6,11,16,21,26,30],'xticklabel',[30,25,20,15,10,5,1],'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
xlim([0,31]);ylim([-0.012,0.005]);
yyaxis right; set(gca,'ycolor','none')
text(ax1,'string',"a",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax2=axes('Position',pos2);
s1 = pcolor(ax2,lon,lat,matrix_ratio_pet);
s1.EdgeColor = 'none'; colormap(ax2,cmap_ratio_pet);hold on
geoshow(ax2,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax2,'string',"d",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.88 0.74 0.015 0.22]);
yy=1:9;imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_igCopy(1:9,:)));ylabel('Lag IG of radiation (\times10^{-3})','FontSize',fsize-1)
set(gca,'ytick',1.5:8.5,'yticklabel',-1.5:-0.5:-5,'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ************** tas ig ************** 
ax3=subplot('Position',pos3);
yyaxis left
plot(ax3,[1,30],[0,0],'--k','linewidth',2);hold on
uper = ts_tas_spring(end-29:end,1)+ts_tas_spring_std(end-29:end,1);    down = ts_tas_spring(end-29:end,1)-ts_tas_spring_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax3,x_patch,y_patch,color_spring,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_tas_summer(end-29:end,1)+ts_tas_summer_std(end-29:end,1);    down = ts_tas_summer(end-29:end,1)-ts_tas_summer_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax3,x_patch,y_patch,color_summer,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_tas_autumn(end-29:end,1)+ts_tas_autumn_std(end-29:end,1);    down = ts_tas_autumn(end-29:end,1)-ts_tas_autumn_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax3,x_patch,y_patch,color_autumn,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_tas_winter(end-29:end,1)+ts_tas_winter_std(end-29:end,1);    down = ts_tas_winter(end-29:end,1)-ts_tas_winter_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax3,x_patch,y_patch,color_winter,'FaceAlpha',.2,'EdgeAlpha',.0);hold on

h1=plot(ax3,ts_tas_spring(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_spring,'MarkerSize',8);hold on
h2=plot(ax3,ts_tas_summer(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_summer,'MarkerSize',8);hold on
h3=plot(ax3,ts_tas_autumn(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_autumn,'MarkerSize',8);hold on
h4=plot(ax3,ts_tas_winter(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_winter,'MarkerSize',8);hold on
legend([h1,h2,h3,h4],{'Spring';'Summer';'Autumn';'Winter'},'location','southwest')
ylabel('IG of temperature (\times10^{-3})')
set(gca,'color','none','ycolor',[0,0,0],'xtick',[1,6,11,16,21,26,30],'xticklabel',[30,25,20,15,10,5,1],'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
xlim([0,31]);ylim([-0.008,0.002]);
yyaxis right; set(gca,'ycolor','none')
text(ax3,'string',"b",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax4=axes('Position',pos4);
s1 = pcolor(ax4,lon,lat,matrix_ratio_vpd);
s1.EdgeColor = 'none'; colormap(ax4,cmap_ratio_vpd);hold on
geoshow(ax4,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax4,'string',"e",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.88 0.42 0.015 0.22]);
yy=1:9;
imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_igCopy(1:9,:)));ylabel('Lag IG of temperature (\times10^{-3})','FontSize',fsize-1)
set(gca,'ytick',1.5:8.5,'yticklabel',-1.5:-0.5:-5,'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ************** Precipitation ************** 
ax5=subplot('Position',pos5);
yyaxis left
plot(ax5,[1,30],[0,0],'--k','linewidth',2);hold on
uper = ts_pr_spring(end-29:end,1)+ts_pr_spring_std(end-29:end,1);    down = ts_pr_spring(end-29:end,1)-ts_pr_spring_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax5,x_patch,y_patch,color_spring,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_pr_summer(end-29:end,1)+ts_pr_summer_std(end-29:end,1);    down = ts_pr_summer(end-29:end,1)-ts_pr_summer_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax5,x_patch,y_patch,color_summer,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_pr_autumn(end-29:end,1)+ts_pr_autumn_std(end-29:end,1);    down = ts_pr_autumn(end-29:end,1)-ts_pr_autumn_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax5,x_patch,y_patch,color_autumn,'FaceAlpha',.2,'EdgeAlpha',.0);hold on
uper = ts_pr_winter(end-29:end,1)+ts_pr_winter_std(end-29:end,1);    down = ts_pr_winter(end-29:end,1)-ts_pr_winter_std(end-29:end,1);    y_patch = cat(1,uper,flipud(down));
patch(ax5,x_patch,y_patch,color_winter,'FaceAlpha',.2,'EdgeAlpha',.0);hold on

h1=plot(ax5,ts_pr_spring(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_spring,'MarkerSize',8);hold on
h2=plot(ax5,ts_pr_summer(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_summer,'MarkerSize',8);hold on
h3=plot(ax5,ts_pr_autumn(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_autumn,'MarkerSize',8);hold on
h4=plot(ax5,ts_pr_winter(end-29:end,1),'o-k','linewidth',2,'MarkerFaceColor',color_winter,'MarkerSize',8);hold on
legend([h1,h2,h3,h4],{'Spring';'Summer';'Autumn';'Winter'},'location','northwest')
ylabel('IG of precipitation (\times10^{-3})');xlabel('Time lag (day)');
set(gca,'color','none','ycolor',[0,0,0],'xtick',[1,6,11,16,21,26,30],'xticklabel',[30,25,20,15,10,5,1],'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
xlim([0,31]);ylim([-0.0002,0.0012]);
yyaxis right; set(gca,'ycolor','none')
text(ax5,'string',"c",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax6=axes('Position',pos6);
s1 = pcolor(ax6,lon,lat,matrix_ratio_pr);
s1.EdgeColor = 'none'; colormap(ax6,cmap_ratio_pr);hold on
geoshow(ax6,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax6,'string',"f",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.88 0.1 0.015 0.22]);
yy=1:7;
imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_ig_pr(1:7,:)));ylabel('Lag IG of precipitation (\times10^{-3})','FontSize',fsize-1)
set(gca,'ytick',1.5:6.5,'yticklabel',[0.1,0.15,0.2,0.25,0.3,0.35,0.4],'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')%


function [matrix,cmap]=get_color_tas(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input<-5.5)=-5.5;
input(-1.5<input&input<=1)=1;
input(-2<input&input<=-1.5)=2;
input(-2.5<input&input<=-2)=3;
input(-3<input&input<-2.5)=4;
input(-3.5<input&input<=-3)=5;
input(-4<input&input<=-3.5)=6;
input(-4.5<=input&input<=-4)=7;
input(-5<=input&input<=-4.5)=8;
input(-5.5<=input&input<=-5)=9;

input(site_noveg)=10;
input(~mask_CONUS_0125d)=11;

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

function [matrix,cmap]=get_color_rad(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input<-5.5)=-5.5;
input(-1.5<input&input<=1)=1;
input(-2<input&input<=-1.5)=2;
input(-2.5<input&input<=-2)=3;
input(-3<input&input<-2.5)=4;
input(-3.5<input&input<=-3)=5;
input(-4<input&input<=-3.5)=6;
input(-4.5<=input&input<=-4)=7;
input(-5<=input&input<=-4.5)=8;
input(-5.5<=input&input<=-5)=9;

input(site_noveg)=10;
input(~mask_CONUS_0125d)=11;

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

function [matrix,cmap]=get_color_pr(input,mask_CONUS_0125d,color_ig)

[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input>0.4)=0.4;
input(-1<=input&input<=0.1)=11;
input(0.1<input&input<=0.15)=12;
input(0.15<input&input<=0.2)=13;
input(0.2<input&input<0.25)=14;
input(0.25<input&input<=0.3)=15;
input(0.3<input&input<=0.35)=16;
input(0.35<input&input<=0.4)=17;

input(site_noveg)=18;
input(~mask_CONUS_0125d)=19;

input=input-10;
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

function[ts,ts_std]=get_ig_ts_std(input)
ts = squeeze(nanmean(input,[1,2]));ts_std = squeeze(nanstd(input,[],[1,2]));
end
