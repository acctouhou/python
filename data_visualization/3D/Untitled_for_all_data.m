clc;
clear;

%%%%%%%%%%%%%%%%
gd=99;
n=(gd-4)^2;
model=2;
%%%%%%%%%%%%
for i=1:34
filename=['x_fin' num2str(i)]; 
a=importdata(filename,' ',0);
filename=['y_split' num2str(i)]; 
b=importdata(filename,' ',0);
filename=['l_split' num2str(i)]; 
c=importdata(filename,' ',0);

local=c(:,1);
y=int64(local/1000);
x=rem(local,1000);
scatter(y,x)
[xx, yy] = meshgrid(1:95,1:95);

aa(i,1)=sum(wtf(a(:,12),length(x),x,y,-2,5,'S',i,b(:,12),xx,yy,'1'));
aa(i,2)=sum(wtf(a(:,11),length(x),x,y,-1,5,'Permeability',i,b(:,11),xx,yy,'2'));
aa(i,3)=sum(wtf(a(:,10),length(x),x,y,-3,3,"Poisson ratio",i,b(:,10),xx,yy,'3'));
aa(i,4)=sum(wtf(a(:,9),length(x),x,y,-2,5,"Young modulus",i,b(:,9),xx,yy,'4'));

end


