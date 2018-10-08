function [ zi ] = vb( x,y,z,gd,xi,yi)
zi = griddata(x, y, z, xi, yi,'linear');

a=isnan(zi);
zi_nearst = griddata(x, y, z, xi, yi,'nearest');
zi(a)=zi_nearst(a);

%mesh(xi,yi,zi)
%
%figure; mesh(xi, yi, zi); view(2); axis image
end

