function [in,out]=data_process(data,num,nn)
% ����1-num��Ϊ���� ��num+1��Ϊ���
n=length(data)-num-nn;
for i=1:n
    x(i,:)=data(i:i+num+nn-1);
end
in=x(:,1:end-nn);
out=x(:,end-nn+1:end);