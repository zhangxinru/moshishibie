clear
clc
%% ����ovariancancer���ݣ������ӻ�
load ovariancancer.mat
grp1=ones(200,1);
str=['Cancer'];
for i=1:216
   grp1(i)=strcmp(grp(i),str)    
    if(grp1(i))
    grp1(i)=0;%cancerת��Ϊ0
    else
    grp1(i)=1;%normalת��Ϊ1
    end
end
%% ����ѵ�������Ͳ�������
% ����ʵ��ѵ�������Ͳ���������ͬ��Ϊovariancancer���ݵ�ǰ���࣬��ֻѡ��������ǰ2ά����ʵ�顣
X = obs(1:2,1:200);
X = [X;ones(1,200)];%��Ӧ��b
Y = grp1(1:200,1);
%% �ݶ��½������Logistic Regression
W = [1,1,1]';
for i = 1 : 1000
    alphak = btl_search(@my_fun,@my_grad,W,X,Y,0.1,0.5);
    alp(i) = alphak;
    W = W - alphak * my_grad(W,X,Y);
    f(i) = my_fun(W,X,Y);
end
%% ����ʧ��������������
figure;
plot(f)
for i = 1 : 200
    y_test(i) = my_sig(W,X(:,i));
end
%% �����ڵ�һ��ĺ�����ʸ�״ͼ
figure
stem(y_test)
sign = 1;
if sign == 1
figure;
%plot(X(1,101:200),X(2,101:200),'b+')
hold on
%plot(X(1,1:100),X(2,1:100),'r<')
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
plot(X(1,101:200),X(2,101:200),'g+')
hold on
plot(X(1,100),X(2,1:100),'y<')
end