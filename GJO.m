function [Male_Jackal_score,Male_Jackal_pos,Convergence_curve]=GJO(SearchAgents_no,Max_iter,lb,ub,dim,fobj)

%% initialize Golden jackal pair
Male_Jackal_pos=zeros(1,dim);
Male_Jackal_score=inf; 
Female_Jackal_pos=zeros(1,dim);  
Female_Jackal_score=inf; 

%% Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iter);

l=0;% Loop counter

% Main loop
while l<Max_iter
        for i=1:size(Positions,1)  

           % boundary checking 上下界确认
            Flag4ub=Positions(i,:)>ub;
            Flag4lb=Positions(i,:)<lb;
            Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               

            % Calculate objective function for each search agent 计算目标函数
            fitness=fobj(Positions(i,:));

            % Update Male Jackal 更新雄性金豺位置
            if fitness<Male_Jackal_score 
                Male_Jackal_score=fitness; 
                Male_Jackal_pos=Positions(i,:);
            end  
             if fitness>Male_Jackal_score && fitness<Female_Jackal_score 
                Female_Jackal_score=fitness; 
                Female_Jackal_pos=Positions(i,:);
            end
        end
    
    
    
     E1=1.5*(1-(l/Max_iter));
       RL=0.05*levy(SearchAgents_no,dim,1.5);%莱维飞行
     
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            E0=2*r1-1;            
           E=E1*E0; % Evading energy
            
            
             if abs(E)<1
                %% EXPLOITATION
               D_male_jackal=abs((RL(i,j)*Male_Jackal_pos(j)-Positions(i,j))); 
                Male_Positions(i,j)=Male_Jackal_pos(j)-E*D_male_jackal;
                D_female_jackal=abs((RL(i,j)*Female_Jackal_pos(j)-Positions(i,j))); 
                Female_Positions(i,j)=Female_Jackal_pos(j)-E*D_female_jackal;
                
            else
                %% EXPLORATION
               D_male_jackal=abs( (Male_Jackal_pos(j)- RL(i,j)*Positions(i,j)));
                Male_Positions(i,j)=Male_Jackal_pos(j)-E*D_male_jackal;
              D_female_jackal=abs( (Female_Jackal_pos(j)- RL(i,j)*Positions(i,j)));
                Female_Positions(i,j)=Female_Jackal_pos(j)-E*D_female_jackal;
             end
             Positions(i,j)=(Male_Positions(i,j)+Female_Positions(i,j))/2;

                                
        end
    end
    l=l+1;    
        
    Convergence_curve(l)=Male_Jackal_score;
     disp(['第',num2str(l),'次测试集的平均相对百分误差为：',num2str(Male_Jackal_score)])
end




