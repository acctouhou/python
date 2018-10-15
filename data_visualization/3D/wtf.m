function [aa] = wtf(z,len,x,y,a,b,name,i,zz,xx,yy,gg)
z_1=zeros([95,95]);
for tt=1:len
    z_1(x(tt),y(tt))=z(tt);
end
mesh(xx,yy, z_1);
view(300,420);
axis([-inf, inf,-inf, inf,a,b]);
title([name ' prediction' num2str(i)]);
set(gca,'fontsize',30)
pname=[gg '_P' num2str(i) '.png']; 
saveas(gcf,pname);

z_11=zeros([95,95]);
for tt=1:length(x)
    z_11(x(tt),y(tt))=zz(tt);
end
mesh(xx,yy, z_11);
axis([-inf, inf,-inf, inf,a,b]);
view(300,420);
title([name ' correct' num2str(i)]);
pname=[gg '_t' num2str(i) '.png']; 
set(gca,'fontsize',30)
saveas(gcf,pname);
error1=(z_11-z_1);
aa=sum(abs(error1));
mesh(xx,yy,error1);
axis([-inf, inf,-inf, inf,-1,1]);
view(300,420);
title([name ' error' num2str(i)]);
pname=[gg '_e' num2str(i) '.png']; 
set(gca,'fontsize',30)
saveas(gcf,pname);
end

