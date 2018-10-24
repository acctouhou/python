clc;
clear;

%%%%%%%%%%%%%%%%
gd=99;
n=45*225;
model=1;
%%%%%%%%%%%%
for yy=1:45
    for zz=1:225
        local(yy,zz)=yy*1000+zz;
    end
end
local=reshape(local,[n,1]);
for i=1:1
filename=['iteration/Iteration' num2str(i) '.inp']; 
a=importdata(filename,',',0);
filename=['diffusion_iteration/Diff_Iteration' num2str(i) '.inp']; 
b=sortrows(importdata(filename,',',0));
filename=['ID_E_P_PERM/ID_e_p_perm_' num2str(i) '.txt']; 
c=sortrows(importdata(filename,',',0));
filename=['S_DATA/S_DATA_' num2str(i) '.txt']; 
d=sortrows(importdata(filename,',',0));


x=a(:,2);
y=a(:,3);
%scale=(max(x)-min(x))/(max(y)-min(y));
%tx=(max(x)-min(x))/gd;
ty=(max(y)-min(y))/225;
tx=ty;
[xi, yi] = meshgrid(min(x):tx:max(x),min(y):ty:max(y));

z_1=a(:,4);%X_displacement  
z_1=vb(x,y,z_1,gd,xi,yi);
mesh(xi,yi,z_1)
axis equal
view(20,490);
title(['X displacement' num2str(i)]);
pname=['1_' num2str(i) '.png']; 
saveas(gcf,pname);
z_1=reshape(z_1(1:end-5,6:end),[n,1]);

z_2=a(:,5); %Y_displacement  
z_2=vb(x,y,z_2,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_2(1:end-5,6:end))
view(20,490)
title(['Y displacement' num2str(i)]);
pname=['2_' num2str(i) '.png']; 
saveas(gcf,pname);
z_2=reshape(z_2(1:end-5,6:end),[n,1]);

z_3=a(:,6); %pressure
z_3=vb(x,y,z_3,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_3(1:end-5,6:end))
view(20,490)
title(['pressure' num2str(i)]);
pname=['3_' num2str(i) '.png']; 
saveas(gcf,pname);
z_3=reshape(z_3(1:end-5,6:end),[n,1]);

z_4=a(:,7); %x_strain 
z_4=vb(x,y,z_4,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_4(1:end-5,6:end))
view(20,490)
title(['x strain' num2str(i)]);
pname=['4_' num2str(i) '.png'];  
saveas(gcf,pname);
z_4=reshape(z_4(1:end-5,6:end),[n,1]);

z_5=a(:,8); %Y_strain
z_5=vb(x,y,z_5,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_5(1:end-5,6:end))
view(20,490)
title(['y strain' num2str(i)]);
pname=['5_' num2str(i) '.png']; 
saveas(gcf,pname);
z_5=reshape(z_5(1:end-5,6:end),[n,1]);

z_10=a(:,14);%X_flow 
z_10=vb(x,y,z_10,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_10(1:end-5,6:end))
view(20,490)
title(['x flow' num2str(i)]);
pname=['10_' num2str(i) '.png']; 
saveas(gcf,pname);
z_10=reshape(z_10(1:end-5,6:end),[n,1]);

z_11=a(:,15); %Y_flow
z_11=vb(x,y,z_11,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_11(1:end-5,6:end))
view(20,490)
title(['y flow' num2str(i)]);
pname=['11_' num2str(i) '.png']; 
saveas(gcf,pname);
z_11=reshape(z_11(1:end-5,6:end),[n,1]);


z_12=b(:,2);%concentration 
z_12=vb(x,y,z_12,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_12(1:end-5,6:end))
view(20,490)
title(['concentration' num2str(i)]);
pname=['12_' num2str(i) '.png'];  
saveas(gcf,pname);
z_12=reshape(z_12(1:end-5,6:end),[n,1]);


z_13=c(:,2);
z_13=vb(x,y,z_13,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_13(1:end-5,6:end))
view(20,490)
title(['YM ' num2str(i)]);
pname=['13_' num2str(i) '.png']; 
saveas(gcf,pname);
z_13=reshape(z_13(1:end-5,6:end),[n,1]);

%*10^6
z_14=c(:,3);%Poisson's ratio
z_14=vb(x,y,z_14,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_14(1:end-5,6:end))
view(20,490)
title(['Poisson ratio ' num2str(i)]);
pname=['14_' num2str(i) '.png']; 
saveas(gcf,pname);
z_14=reshape(z_14(1:end-5,6:end),[n,1]);

%*10^15
z_15=c(:,4);%Permeability
z_15=vb(x,y,z_15,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_15(1:end-5,6:end))
view(20,490)
title(['Permeability ' num2str(i)]);
pname=['15_' num2str(i) '.png']; 
saveas(gcf,pname);
z_15=reshape(z_15(1:end-5,6:end),[n,1]);

z_16=d(:,2);
z_16(length(d):length(c))=0;
z_16=vb(x,y,z_16,gd,xi,yi);
mesh(xi(1:end-5,6:end),yi(1:end-5,6:end),z_16(1:end-5,6:end))
view(20,490)
title(['S' num2str(i)]);
pname=['16_' num2str(i) '.png']; 
saveas(gcf,pname);
z_16=reshape(z_16(1:end-5,6:end),[n,1]);


filename=num2str(i); 
fid =fopen(filename, 'w'); 

for tt=1:n
    fprintf(fid,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d',z_1(tt),z_2(tt),z_3(tt),z_4(tt),z_5(tt),z_10(tt),z_11(tt),z_12(tt),z_13(tt),z_14(tt),z_15(tt),z_16(tt),local(tt),model,i);
    fprintf(fid,'\n');
end

end


