load('../../data/missIdx.mat');
load('../../data/provideIdx.mat');
load('../../data/Train.mat');

mx = x(missIdx);
my = y(missIdx);
mz = z(missIdx);
px = x(provideIdx);
py = y(provideIdx);
pz = z(provideIdx);

scatter3(mx,my,mz,'b');
hold on
scatter3(px,py,pz,'r.');