% ** README **
% Qian Zhao, 08/24/2025
% This script is for reproducing the Extended Figure 2 of the paper 'Deep learning for flash drought prediction and interpretation'
% The data related to IT-Drought output were provided in Zenodo https://zenodo.org/uploads/16903004
% The climate data were not provided due to lack of license, but you can directly download from their websites
% The auxiliary data were not provided here, including masks and colorbar
% Welcome to cite our paper and Zenodo. 
% Please contact the corresponding author if any questions

%% Identify flash droughts
aggstep = 7; up_threshold=0.4;   dn_threshold=0.2;

load('SMroot_1979_2022.mat');
SMobs_pend=aggregate2weekly(SMroot,aggstep);
SMobs_percentile=get_percentile(SMobs_pend);
[rws,cls,peds,yrs]=size(SMobs_percentile);
wks=peds*yrs;

NLDASroot_droughtBinary_FD  = nan(rws,cls,wks);
NLDASroot_drought_FD= nan(rws,cls,20,6);

for ir = 1:rws
    for ic = 1:cls
        SM_percentile_ts = squeeze(SMobs_percentile(ir,ic,:,:));

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
                    NLDASroot_drought_FD(ir,ic,1:i,:)= FDdrought;
                    NLDASroot_droughtBinary_FD(ir,ic,:)=FDdroughtBinary;
                end
            end
        end
    end
end

%% plot figure
load('color_FDduration.mat');load('mask_CONUS_0125d.mat')
temp= NLDASroot_drought_FD(:,:,:,1);temp2 = nansum(temp,3)/44;
[matrix_ratio_NLDASroot,cmap_ratio_NLDASroot] = Get_figure_input(temp2,mask_CONUS_0125d,color_FDduration);

filepath = 'cb_2018_us_state_20m.shp';Map = shaperead(filepath);Map([8,26,49])=[];
load('LatLon_Grid0125d.mat');lon = Lon_grid0125d(1,:);lat = Lat_grid0125d(:,1);

fig = figure('color','white');set(gcf,'Units','centimeters','position',[15 15 20 14]);pos1=[0.15 0.2 0.7 0.65];

ax1=axes('Position',pos1,'Layer','top');
s4 = pcolor(ax1,lon,lat,matrix_ratio_NLDASroot);
s4.EdgeColor = 'none'; colormap(ax1,cmap_ratio_NLDASroot);hold on
geoshow(ax1,Map,'FaceColor','none','facealpha',1, 'EdgeColor','k');
set(gca,'ytick',[26,36,46],'yticklabel',{'26\circN','36\circN','46\circN'},'xtick',[-120,-100,-80],...
    'xticklabel',{'120\circW','100\circW','80\circW'},'FontSize',fsize,'FontName','Arial','box','on');

ax1=axes('position',[0.87 0.2 0.025 0.65]);
yy=1:7;yy = yy';
imagesc(ax1,flipud(yy))
colormap(ax1,color_FDduration(1:7,:));
xlabel('times/year')
set(gca,'ytick',1.5:6.5,'yticklabel',[1.2,1.0,0.8,0.6,0.4,0.2],'xtick',[],'FontSize',fsize,'FontName','Arial','yaxislocation','right')


function [matrix,cmap]=Get_figure_input(input,mask_CONUS,colors)

[rws,cls]=size(input);
site_noveg =mask_CONUS& isnan(input);
input(input<0)=0;input(input>2)=2;

input(0<=input&input<=0.2)=11;
input(0.2<input&input<=0.4)=12;
input(0.4<input&input<=0.6)=13;
input(0.6<input&input<=0.8)=14;
input(0.8<input&input<=1.0)=15;
input(1.0<input&input<=1.2)=16;
input(1.2<input&input<=2)=17;
input(site_noveg)=18;
input(~mask_CONUS)=19;

[indx,indy]=find(mask_CONUS==1);
numb=1:length(indx);
matrix=nan(rws,cls);
for i=1:length(indx)
    matrix(indx(i),indy(i))=numb(i);
end

input(~mask_CONUS)=nan;
dmap = nan(rws,cls,3);
for ir =1:rws
    for ic = 1:cls
        id = input(ir,ic)-10;
        if ~isnan(id)
            dmap(ir,ic,:) = colors(id,:);
        end
    end
end

cmap = reshape(dmap,[rws*cls,3]);
site = find(isnan(cmap(:,3)));
cmap(site,:)=[];
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
