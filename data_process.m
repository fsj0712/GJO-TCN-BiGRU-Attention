function [in,out]=data_process(data,num,nn)
% 采用1-num作为输入 第num+1作为输出
n=length(data)-num-nn;
for i=1:n
    x(i,:)=data(i:i+num+nn-1);
end
in=x(:,1:end-nn);
out=x(:,end-nn+1:end);