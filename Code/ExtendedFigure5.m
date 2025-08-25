% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Extended Figure 5 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
load('IG_SMsurf.mat');load('IG_SMroot.mat');load('IG_windspeed.mat');load('IG_snow.mat');load('IG_surfacepressure.mat');load('IG_VPD.mat')
load('color_vartime.mat');load('mask_CONUS_0125d.mat');load('color_blue2red_2.mat');load('color_relation_SMM_SM.mat');load('color_ig_0112Copy_pr.mat');load('color_ig_0112Copy.mat')

pattern_SMroot = nanmean(ig_onset_FD_pattern_SMroot(:,:,end-6:end,1),3);
pattern_SMsurf = nanmean(ig_onset_FD_pattern_SMsurf(:,:,end-6:end,1),3);
pattern_vs = nanmean(ig_onset_FD_pattern_vs(:,:,end-6:end,1),3);
pattern_snow = nanmean(ig_onset_FD_pattern_snow(:,:,end-6:end,1),3);
pattern_sp = nanmean(ig_onset_FD_pattern_sp(:,:,end-6:end,1),3);
pattern_vpd = nanmean(ig_onset_FD_pattern_vpd(:,:,end-6:end,1),3);

[matrix_ratio_SMroot,cmap_ratio_SMroot] = get_color_SMroot(pattern_SMroot,mask_CONUS_0125d,color_ig_pr);
[matrix_ratio_SMsurf,cmap_ratio_SMsurf] = get_color_SMsurf(pattern_SMsurf*1000,mask_CONUS_0125d,color_ig_pr);
[matrix_ratio_vs,cmap_ratio_vs] = get_color_vs(pattern_vs*1000,mask_CONUS_0125d,color_ig_pr);
[matrix_ratio_snow,cmap_ratio_snow] = get_color_snow(pattern_snow*10000,mask_CONUS_0125d,color_ig_pr);
[matrix_ratio_sp,cmap_ratio_sp] = get_color_sp(pattern_sp*1000,mask_CONUS_0125d,color_igCopy([1:7,11,12],:));
[matrix_ratio_vpd,cmap_ratio_vpd] = get_color_vpd(pattern_vpd*1000,mask_CONUS_0125d,color_igCopy([1:7,11,12],:));

%% plot figure
filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);x = 1:30;x_patch = cat(2,x,fliplr(x));

fig = figure('color','white');fsize = 12;
set(gcf,'Units','centimeters','Position',[3 1 24 22])

pos1=[0.06 0.74 0.34 0.22];pos2=[0.55 0.74 0.34 0.22];pos3=[0.06 0.42 0.34 0.22];
pos4=[0.55 0.42 0.34 0.22];pos5=[0.06 0.1 0.34 0.22];pos6=[0.55 0.1 0.34 0.22];
pos_label = [-0.09,1.1]; pos_varname = [0.45,1.06];

% **************  SM rootzone ************** 
ax1=subplot('Position',pos1);
s1 = pcolor(ax1,lon,lat,matrix_ratio_SMroot);
s1.EdgeColor = 'none'; colormap(ax1,cmap_ratio_SMroot);hold on
geoshow(ax1,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax1,'string',"a",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.405 0.74 0.015 0.22]);yy=1:7;imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_ig_pr(1:7,:)));ylabel('Lag IG of rootzone SM','FontSize',fsize-1)
set(gca,'ytick',1.5:6.5,'yticklabel',[0.01,0.02,0.025,0.03,0.035,0.04],'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% **************  SM surface ************** 
ax2=axes('Position',pos2);
s1 = pcolor(ax2,lon,lat,matrix_ratio_SMsurf);
s1.EdgeColor = 'none'; colormap(ax2,cmap_ratio_SMsurf);hold on
geoshow(ax2,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax2,'string',"b",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.895 0.74 0.015 0.22]);yy=1:7;imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_ig_pr(1:7,:)));ylabel('Lag IG of surface SM (\times10^{-3})','FontSize',fsize-1)
set(gca,'ytick',1.5:6.5,'yticklabel',2:1:7,'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ************** wind speed ************** 
ax3=subplot('Position',pos3);
s1 = pcolor(ax3,lon,lat,matrix_ratio_vs);
s1.EdgeColor = 'none'; colormap(ax3,cmap_ratio_vs);hold on
geoshow(ax3,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax3,'string',"c",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.405 0.42 0.015 0.22]);yy=1:7;imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_ig_pr(1:7,:)));ylabel('Lag IG of wind speed (\times10^{-3})','FontSize',fsize-1)
set(gca,'ytick',1.5:6.5,'yticklabel',0.1:0.1:0.6,'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ************** snow ************** 
ax4=subplot('Position',pos4);
s1 = pcolor(ax4,lon,lat,matrix_ratio_snow);
s1.EdgeColor = 'none'; colormap(ax4,cmap_ratio_snow);hold on
geoshow(ax4,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax4,'string',"d",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.895 0.42 0.015 0.22]);yy=1:7;imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_ig_pr(1:7,:)));ylabel('Lag IG of snow (\times10^{-4})','FontSize',fsize-1)
set(gca,'ytick',1.5:6.5,'yticklabel',[0.02,0.04,0.06,0.08,0.1,0.3],'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ************** surface pressure ************** 
ax5=subplot('Position',pos5);
s1 = pcolor(ax5,lon,lat,matrix_ratio_sp);
s1.EdgeColor = 'none'; colormap(ax5,cmap_ratio_sp);hold on
geoshow(ax5,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax5,'string',"e",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.405 0.1 0.015 0.22]);yy=1:7;imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_igCopy(1:7,:)));ylabel('Lag IG of surface pressure (\times10^{-3})','FontSize',fsize-1)
set(gca,'ytick',1.5:6.5,'yticklabel',[-0.6,-0.9,-1.2,-1.4,-1.6,-1.8],'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ************** vpd ************** 
ax6=subplot('Position',pos6);
s1 = pcolor(ax6,lon,lat,matrix_ratio_vpd);
s1.EdgeColor = 'none'; colormap(ax6,cmap_ratio_vpd);hold on
geoshow(ax6,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax6,'string',"f",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.895 0.1 0.015 0.22]);yy=1:7;imagesc(ax,flipud(yy'))
colormap(ax,flipud(color_igCopy(1:7,:)));ylabel('Lag IG of VPD (\times10^{-3})','FontSize',fsize-1)
set(gca,'ytick',1.5:6.5,'yticklabel',[-0.1,-0.15,-0.2,-0.25,-0.3,-0.35],'xticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')


function [matrix,cmap]=get_color_SMroot(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input>0.05)=0.05;
input(-1<=input&input<=0.01)=11;
input(0.01<input&input<=0.02)=12;
input(0.02<input&input<=0.025)=13;
input(0.025<input&input<0.03)=14;
input(0.03<input&input<=0.035)=15;
input(0.035<input&input<=0.04)=16;
input(0.04<input&input<=0.05)=17;

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

function [matrix,cmap]=get_color_SMsurf(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input>8)=8;
input(0<=input&input<=2)=11;
input(2<input&input<=3)=12;
input(3<input&input<=4)=13;
input(4<input&input<5)=14;
input(5<input&input<=6)=15;
input(6<input&input<=7)=16;
input(7<input&input<=8)=17;

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

function [matrix,cmap]=get_color_vs(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input>0.7)=0.7;
input(0<=input&input<=0.1)=11;
input(0.1<input&input<=0.2)=12;
input(0.2<input&input<=0.3)=13;
input(0.3<input&input<0.4)=14;
input(0.4<input&input<=0.5)=15;
input(0.5<input&input<=0.6)=16;
input(0.6<input&input<=0.7)=17;

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

function [matrix,cmap]=get_color_snow(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input>0.4)=0.4;
input(-0.2<input&input<=0.02)=11;
input(0.02<input&input<=0.04)=12;
input(0.04<input&input<0.06)=13;
input(0.06<input&input<=0.08)=14;
input(0.08<input&input<=0.1)=15;
input(0.1<input&input<=0.3)=16;
input(0.3<input&input<=0.4)=17;

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

function [matrix,cmap]=get_color_sp(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input<-2)=-2;
input(-0.6<input&input<=1)=1;
input(-0.9<input&input<=-0.6)=2;
input(-1.2<input&input<=-0.9)=3;
input(-1.4<input&input<-1.2)=4;
input(-1.6<input&input<=-1.4)=5;
input(-1.8<input&input<=-1.6)=6;
input(-2<=input&input<=-1.8)=7;

input(site_noveg)=8;
input(~mask_CONUS_0125d)=9;

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

function [matrix,cmap]=get_color_vpd(input,mask_CONUS_0125d,color_ig)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input<-0.5)=-0.5;
input(-0.1<input&input<=0.2)=1;
input(-0.15<input&input<=-0.1)=2;
input(-0.2<input&input<=-0.15)=3;
input(-0.25<input&input<-0.2)=4;
input(-0.3<input&input<=-0.25)=5;
input(-0.35<input&input<=-0.3)=6;
input(-0.5<=input&input<=-0.35)=7;

input(site_noveg)=8;
input(~mask_CONUS_0125d)=9;

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