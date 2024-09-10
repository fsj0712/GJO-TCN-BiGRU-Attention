
classdef FlipLayer < nnet.layer.Layer
%%  数据翻转
    methods
        function layer = FlipLayer(name)
            layer.Name = name;
        end
        function Y = predict(~, X)
                 Y = flip(X, 3);
        end
    end
end
