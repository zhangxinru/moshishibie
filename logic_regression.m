clear
clc
%% 读入ovariancancer数据，并可视化
load ovariancancer.mat
grp1=ones(200,1);
str=['Cancer'];
for i=1:216
   grp1(i)=strcmp(grp(i),str)    
    if(grp1(i))
    grp1(i)=0;%cancer转换为0
    else
    grp1(i)=1;%normal转换为1
    end
end
%% 
% iris数据包含三类，每类50个样本共150个，每个样本的维数是4。用PCA降到2维进行可视化。iris数据的第一类与其它两类分的比较开，第二类和第三类有重叠。
%[Trans,~] = pca(irisInputs');
%iris_dr = Trans'*irisInputs;
%figure;
%plot(iris_dr(1,:),iris_dr(2,:),'.','MarkerSize',10);
%% 构造训练样本和测试样本
% 本次实验训练样本和测试样本相同，为ovariancancer数据的前两类，并只选用样本的前2维进行实验。
X = obs(1:2,1:200);
X = [X;ones(1,200)];%对应于b
Y = grp1(1:200,1);
%% 梯度下降法求解Logistic Regression
W = [1,1,1]';
for i = 1 : 1000
    alphak = btl_search(@my_fun,@my_grad,W,X,Y,0.1,0.5);
    alp(i) = alphak;
    W = W - alphak * my_grad(W,X,Y);
    f(i) = my_fun(W,X,Y);
end
%% 画损失函数的收敛曲线
figure;
plot(f)
for i = 1 : 100
    y_test(i) = my_sig(W,X(:,i));
end
%% 画属于第一类的后验概率杆状图
figure
stem(y_test)
sign = 1;
if sign == 1
figure;
%plot(X(1,51:100),X(2,51:100),'b+')
hold on
%plot(X(1,1:50),X(2,1:50),'r<')
for i = 4:0.01:7
    for j = 2:0.01:4.5
        a = [i;j;1];
        if my_sig(W,a)>0.5
            plot(i,j,'b.','MarkerSize',20);
        else
            plot(i,j,'r.','MarkerSize',20);
        end
    end
end
plot(X(1,51:100),X(2,51:100),'g+')
hold on
plot(X(1,1:50),X(2,1:50),'y<')
end