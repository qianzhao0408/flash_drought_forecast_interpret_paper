% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Extended Figure 3 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% load and prepare data
load('SlopeSpectrum.mat');load('ACC_ITDrought.mat');load('color_blue2red.mat');load('mask_CONUS_0125d.mat');load('color_relation_SMM_SM.mat');load('SMroot_1979_2022.mat')

[matrix_ratio_spectrum,cmap_ratio_spectrum] = get_color_spectrum(slope_spectrum_log10,mask_CONUS_0125d,color_blue2red([2:9,11:12],:));
ACC_pattern = squeeze(nanmean(DL_ACC_daily_twotest(:,:,1:14),3));
ACC_points = ACC_pattern(:);spectrum_points = slope_spectrum_log10(:);
site = find(isnan(ACC_points)|isnan(spectrum_points));
ACC_points(site)=[]; spectrum_points(site)=[];
stats_ACC= regstats(ACC_points,spectrum_points,'linear');

Y_pred = stats_ACC.yhat;
data = [spectrum_points,ACC_points];
density_2D = ksdensity([data(:,1),data(:,2)],[data(:,1),data(:,2)]);

% panel a-d
f_low = 1/365; f_high = 1/7;

pot1 = [103,52];ir = pot1(1); ic = pot1(2);ts = squeeze(SMroot(ir,ic,:,:));  ts=ts(:);
[f1,A1,~]=Cal_spectrum(ts);

site_range1 = find(f1>=f_low&f1<=f_high);
f1_range = f1(site_range1);            A1_range = A1(site_range1);
f1_log = log10(f1_range);            A1_log = log10(A1_range);

stats1 = regstats(A1_log,f1_log,'linear'); trend_A1 = stats1.tstat.beta(2);
pValue1 =stats1.tstat.pval(2);inter1 = stats1.tstat.beta(1);
ts_linear1 = f1_log*trend_A1+inter1;

pot2 = [120,337];ir2 = pot2(1); ic2 = pot2(2);ts2 = squeeze(SMroot(ir2,ic2,:,:));  ts2=ts2(:);
[f2,A2,~]=Cal_spectrum(ts2);

site_range2 = find(f2>=f_low&f2<=f_high);
f2_range = f2(site_range2);            A2_range = A2(site_range2);
f2_log = log10(f2_range);            A2_log = log10(A2_range);

stats2 = regstats(A2_log,f2_log,'linear'); trend_A2 = stats2.tstat.beta(2);
pValue2 =stats2.tstat.pval(2);inter2 = stats2.tstat.beta(1);
ts_linear2 = f2_log*trend_A2+inter2;

%% plot figure
filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);

fig = figure('color','white');set(gcf,'Units','centimeters','Position',[3 3 30 28]);
hori = 0.22; wid = 0.36; fsize = 17; pos_label = [-0.18,1.05];

pos_a=[0.09 0.75 wid hori];pos_b=[0.59 0.75 wid hori];
pos_c=[0.09 0.45 wid hori];pos_d=[0.59 0.45 wid hori];
pos_5=[0.09 0.15 wid hori];pos_6=[0.59 0.15 wid hori];

ax1=axes('Position',pos_a);
title(ax1,'Time series of SM','fontsize',18)
plot(ax1,ts(:)/1000);hold on
annotation('arrow', [0.42, 0.5], [0.94, 0.94]);
ylabel('SM (m^3 m^{-3})');xlim([1,365*44])
xlabel('Year');
set(gca,'xtick',1:365*10:365*44,'xticklabel',1979:10:2022,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');%'xtick',1:5:46,'ytick',0:0.2:1,
text(ax1,'string',"a",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');
text(ax1,'string',"fft",'Units','normalized','position',[1,0.95],'fontsize',fsize+2,'FontName','Arial');

ax2=axes('Position',pos_b);
title(ax2,'Power spectrum','fontsize',18)
plot(ax2,f1_log,A1_log);hold on
plot(ax2,f1_log,ts_linear1,'-r')
xlim([-2.6,-0.8]);ylim([1,6])
text(-2.3,2,['slope=',num2str(trend_A1,'%.2f')],'fontsize',18)
xlabel('log_{10}(Frequency (day^{-1}))');ylabel('log_{10}(power spectrum)')
set(gca,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');%'xtick',1:5:46,'ytick',0:0.2:1,
text(ax2,'string',"b",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax3=axes('Position',pos_c);
title(ax3,'Time series of SM','fontsize',18)
plot(ax3,ts2(:)/1000);hold on
annotation('arrow', [0.42, 0.5], [0.64, 0.64]);
ylabel('SM (m^3 m^{-3})');xlim([1,365*44])
xlabel('Year');
set(gca,'xtick',1:365*10:365*44,'xticklabel',1979:10:2022,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');%'xtick',1:5:46,'ytick',0:0.2:1,
text(ax3,'string',"c",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');
text(ax3,'string',"fft",'Units','normalized','position',[1,0.95],'fontsize',fsize+2,'FontName','Arial');

ax4=axes('Position',pos_d);
title(ax4,'Power spectrum','fontsize',18)
plot(ax4,f2_log,A2_log);hold on
plot(ax4,f2_log,ts_linear2,'-r')
xlim([-2.6,-0.8]);ylim([1,6])
text(-2.3,2,['slope=',num2str(trend_A2,'%.2f')],'fontsize',18)
xlabel('log_{10}(Frequency (day^{-1}))');ylabel('log_{10}(power spectrum)')
set(gca,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');%'xtick',1:5:46,'ytick',0:0.2:1,
text(ax4,'string',"d",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

% ---------------------- panel e ---------------------- 
ax5=axes('Position',pos_5);
s1 = pcolor(ax5,lon,lat,matrix_ratio_spectrum);
s1.EdgeColor = 'none'; colormap(ax5,cmap_ratio_spectrum);hold on
geoshow(ax5,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');hold on
plot(ax5,lon(ic),lat(ir),'.','Markersize',20,'color',[1.00,0.00,0.00]);hold on
plot(ax5,lon(ic2),lat(ir2),'.','Markersize',20,'color',[0,1,0]);hold on
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');
text(ax5,'string',"e",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');

ax=axes('position',[0.09 0.09 wid 0.011]); yy=1:8;imagesc(ax,yy)
colormap(ax,color_blue2red(2:9,:));xlabel('Power spectrum exponent')
set(gca,'xtick',1.5:1:7.5,'xticklabel',-1.2:0.1:-0.6,'yticklabel',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')

% ---------------------- panel f ---------------------- 
ax6=axes('Position',pos_6);
scatter(ax6,data(:,1),data(:,2),5,density_2D,'filled');
colormap(ax6,flipud(map_listCopy)/255);hold on;
plot(ax6,spectrum_points, Y_pred,'-','linewidth',2,'color',[0.85,0.33,0.10]);
xlabel('Power spectrum exponent');ylabel('ACC');
ylim([0.3,1]);xlim([-1.45,-0.5]);
set(gca,'fontsize',fsize,'linewidth',1.5,'FontName','Arial','box','off');
text(ax6,'string',"f",'Units','normalized','position',pos_label,'fontsize',fsize+2,'FontName','Arial','FontWeight','bold');
text(ax6,-1.2,0.5,'p','fontsize',fsize,'fontangle','italic')
text(ax6,-1.17,0.5,'<0.001 ','fontsize',fsize)


function [matrix,cmap]=get_color_spectrum(input,mask_CONUS_0125d,color_bar)
[rws,cls]=size(input);
site_noveg =mask_CONUS_0125d& isnan(input);

input(input<-1.5)=-1.5;input(input>-0.5)=-0.5;
input(-1.5<=input&input<=-1.2)=11;
input(-1.2<input&input<=-1.1)=12;
input(-1.1<input&input<=-1.0)=13;
input(-1.0<input&input<=-0.9)=14;
input(-0.9<input&input<=-0.8)=15;
input(-0.8<input&input<=-0.7)=16;
input(-0.7<input&input<=-0.6)=17;
input(-0.6<=input&input<=-0.5)=18;

input(site_noveg)=19;
input(~mask_CONUS_0125d)=20;
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

function [f,A,x]=Cal_spectrum(input_ts)

x = input_ts(:);
N = length(x);
n=0:N-1;
X = fft(x);

Ts=1;
Fs=1;%
df=Fs/N; 
f=(0:1:N/2)*df;
period = 1./f;

Y=X(1:N/2+1);          
Y(2:end-1)=2*Y(2:end-1);

A=abs(Y);  
clc
end