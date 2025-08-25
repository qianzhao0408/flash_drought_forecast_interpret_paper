% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Extended Figure 1 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
load('mask_CONUS_0125d.mat');load('mask_CONUS_025d.mat');load('mask_CONUS_1p5d.mat');load('color_blue2red.mat')
load('F1_ITDrought.mat');load('ACC_ITDrought.mat');load('mask_forF1socre.mat')
thr = 0.5;
[tsrecall_DLdaily,Xpatchrecall_DLdaily,Ypatchrecall_DLdaily]=get_uncertainty_fordraw(DL_recall_FD_twotest,thr,mask_forF1socre);
[tsprecision_DLdaily,Xpatchprecision_DLdaily,Ypatchprecision_DLdaily]=get_uncertainty_fordraw(DL_precision_FD_twotest,thr,mask_forF1socre);

recall_pattern = squeeze(nanmean(DL_recall_FD_twotest(:,:,1:14),3));
[matrix_ratio_recall,cmap_ratio_recall] = get_color_acc(recall_pattern,mask_CONUS_0125d,color_blue2red);
precision_pattern = squeeze(nanmean(DL_precision_FD_twotest(:,:,1:14),3));
[matrix_ratio_precision,cmap_ratio_precision] = get_color_acc(precision_pattern,mask_CONUS_0125d,color_blue2red);

%% plot figure

filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);x = 1:30;x_patch = cat(2,x,fliplr(x));

load('color_vartime14.mat');load('color_blue2red.mat');fsize = 17; marksize = 15; linewid = 1.5;
fig = figure('color','white');set(gcf,'Units','centimeters','Position',[3 3 32 24]);
hori = 0.3; wid = 0.38;
pos_a=[0.09 0.60 wid hori+0.03]; pos_b=[0.57 0.60 wid hori+0.03];
pos_c=[0.09 0.19 wid hori-0.017]; pos_d=[0.57 0.19 wid hori-0.017];pos_label = [-0.1,1.04];

% ------------- panel a: precision and recall time series ------------- 
ax3=subplot('Position',pos_a);
patch(ax3,Xpatchrecall_DLdaily,Ypatchrecall_DLdaily,[0.86,0.05,0.20],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h1=plot(ax3,tsrecall_DLdaily,'.-','linewidth',linewid,'markersize',marksize,'color',[0.86,0.05,0.20]);hold on;

patch(ax3,Xpatchprecision_DLdaily,Ypatchprecision_DLdaily,[0.20,0.47,0.81],'FaceAlpha',.1,'EdgeAlpha',.0);hold on
h3=plot(ax3,tsprecision_DLdaily,'.-','linewidth',linewid,'markersize',marksize,'Color',[0.20,0.47,0.81]);hold on;

xlim([0,46]);ylim([0,1])
xlabel('Lead time (day)','FontSize',fsize,'FontName','Arial')
ylabel('Recall and precision','FontSize',fsize,'FontName','Arial')

set(gca,'xtick',1:5:46,'ytick',0:0.2:1,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
legend([h1,h3],{'Recall','Precision'},'NumColumns',1,'fontsize',fsize-2,'edgecolor','none')
text(ax3,'string',"a",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

% ------------- panel b: recall spatial pattern ------------- 
ax2=axes('Position',pos_c);
s1 = pcolor(ax2,lon,lat,matrix_ratio_recall);%title(ax2,'VPD','FontSize',fsize,'FontName','Arial')
s1.EdgeColor = 'none'; colormap(ax2,cmap_ratio_recall);hold on
geoshow(ax2,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax2,'string',"b",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.09 0.1 wid 0.015]);
yy=1:10;%yy = yy;
imagesc(ax,yy)
colormap(ax,color_blue2red(1:10,:));xlabel('Average recall')
set(gca,'xtick',0.5:2:10.5,'xticklabel',0:0.2:1,'yticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')


% ------------- panel c: precision spatial pattern ------------- 
ax4=axes('Position',pos_d);
s1 = pcolor(ax4,lon,lat,matrix_ratio_precision);
s1.EdgeColor = 'none'; colormap(ax4,cmap_ratio_precision);hold on
geoshow(ax4,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax4,'string',"c",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.57 0.1 wid 0.015]);
yy=1:10;%yy = yy;
imagesc(ax,yy)
colormap(ax,color_blue2red(1:10,:));xlabel('Average precision')
set(gca,'xtick',0.5:2:10.5,'xticklabel',0:0.2:1,'yticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')%


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
