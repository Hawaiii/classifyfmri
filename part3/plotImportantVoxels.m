% plot brain change across time for one subject

mx0 = csvread('mx_0.csv');
my0 = csvread('my_0.csv');
mz0 = csvread('mz_0.csv');

mx1 = csvread('mx_1.csv');
my1 = csvread('my_1.csv');
mz1 = csvread('mz_1.csv');

mx3 = csvread('mx_3.csv');
my3 = csvread('my_3.csv');
mz3 = csvread('mz_3.csv');

figure;
hold on
scatter3(mx0, my0, mz0,'r');
scatter3(mx1, my1, mz1,'g');
scatter3(mx3, my3, mz3,'b');