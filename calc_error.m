
%%
function [test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y)


test_MAE=sum(abs(y_test_predict-test_y))/length(test_y) ;
test_MAPE=sum(abs((y_test_predict-test_y)./test_y))/length(test_y);
test_MSE=(sum(((y_test_predict-test_y)).^2)/length(test_y));
test_RMSE=sqrt(sum(((y_test_predict-test_y)).^2)/length(test_y));
test_R2= 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);


disp(['1.均方差(MSE)：',num2str(test_MSE)])
disp(['2.根均方差(RMSE)：',num2str(test_RMSE)])
disp(['3.平均绝对误差（MAE）：',num2str(test_MAE)])
disp(['4.平均相对百分误差（MAPE）：',num2str(test_MAPE*100),'%'])
disp(['5.R2：',num2str(test_R2*100),'%'])
end


