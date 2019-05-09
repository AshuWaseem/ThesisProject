%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) Pedro Antonio Gutiérrez (pagutierrez at uco dot es)
% María Pérez Ortiz (i82perom at uco dot es)
% Javier Sánchez Monedero (jsanchezm at uco dot es)
%
% This file implements the code for the NPSVOR method.
% 
% The code has been tested with Ubuntu 12.04 x86_64, Debian Wheezy 8, Matlab R2009a and Matlab 2011
% 
% If you use this code, please cite the associated paper
% Code updates and citing information:
% http://www.uco.es/grupos/ayrna/orreview
% https://github.com/ayrna/orca
% 
% AYRNA Research group's website:
% http://www.uco.es/ayrna 
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 3
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA. 
% Licence available at: http://www.gnu.org/licenses/gpl-3.0.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef RampKSVCR < Algorithm
    %NPSVOR: Nonparallel Support Vector Ordinal Regression
    
    properties
       
        name_parameters = {'C','e','k','rho','t','s','threshold'} %C:c1=c2=C; k: rbf, e: epsilon,  rho: ADMM, t: Ins_t, s: Hinge_s

        parameters
    end
    
    methods
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: NPSVOR (Public Constructor)
        % Description: It constructs an object of the class
        %               NPSVOR Ordinal and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           kernel--> Type of Kernel function
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = RampKSVCR(kernel)
            obj.name = 'RampKSVCR';
            if(nargin ~= 0)
                 obj.kernelType = kernel;
            else
                obj.kernelType = 'rbf';
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: defaultParameters (Public)
        % Description: It assigns the parameters of the 
        %               algorithm to a default value.
        % Type: Void
        % Arguments: 
        %           No arguments for this function.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function obj = defaultParameters(obj)
            obj.parameters.C = 10.^(-3:1:3);
	    % kernel width
            obj.parameters.k = 10.^(-3:1:3);
            obj.parameters.e = 0.1;
            obj.parameters.rho = 1;
            obj.parameters.t = 2.^(-3:1:3); %[2^-3, 2^-2,...,2^3]
            obj.parameters.s = -2.^(-3:1:3); 
            obj.parameters.threshold = 0; 
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: runAlgorithm (Public)
        % Description: This function runs the corresponding
        %               algorithm, fitting the model and 
        %               testing it in a dataset.
        % Type: It returns the model (Struct) 
        % Arguments: 
        %           Train --> Training data for fitting the model
        %           Test --> Test data for validation
        %           parameters --> vector with the parameter information
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [model_information] = runAlgorithm(obj,train, test, parameters)
                param.C = parameters(1);
                param.k = parameters(2);
                param.e = parameters(3);
                param.t = parameters(4);
                param.s = parameters(5);
                param.threshold=parameters(6);
                param.rho= 1;
                train.uniqueTargets = unique([train.targets]);
                test.uniqueTargets = train.uniqueTargets;
                train.nOfClasses = length(train.uniqueTargets);
                test.nOfClasses = train.nOfClasses;
                train.nOfPatterns = length(train.targets);
                test.nOfPatterns = length(test.targets);                
                
                c1 = clock;
                model = obj.train(train, param);
                c2 = clock;
                model_information.trainTime = etime(c2,c1);
                
                c1 = clock;
                [model_information.projectedTrain, model_information.predictedTrain] = obj.test(train,train, model,param);
                [model_information.projectedTest,model_information.predictedTest ] = obj.test(train, test,model,param);
                c2 = clock;
                model_information.testTime = etime(c2,c1);
                           
                model.algorithm = 'RampKSVCR';
                model.parameters = param;
                model_information.model = model;

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: train (Public)
        % Description: This function train the model for
        %               the NPSVOR algorithm.
        % Type: It returns the model
        % Arguments: 
        %           train --> Train struct
        %           param--> struct with the parameter information
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        function [model]= train( obj, train, param) 
            model = RampLossKSVCR(train.patterns,train.targets,param.C,param.C,param.e,param.rho,param.t, param.s, obj.kernelType, param.k);
            model.labelSet = unique(train.targets);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: test (Public)
        % Description: This function test a model given in
        %               a set of test patterns.
        % Outputs: Two arrays (decision values and predicted targets)
        % Arguments: 
        %           test --> Test struct data
        %           model --> struct with the model information
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [decv, pred]= test(obj, train, test,model,param)
          nT=test.nOfPatterns;  
          decv= zeros(nT,length(model.b));     
          pred= zeros(nT,length(model.b));   
          if nT >1000
              for j=1:nT/1000
                  xnewk=test.patterns(1000*(j-1)+1:1000*j,:);
                  decv(1000*(j-1)+1:1000*j,:) = Kernel(obj.kernelType, xnewk',train.patterns',param.k)* model.Beta+ ones(size(xnewk,1),1)*model.b;
              end
              xnewk=test.patterns(1000*j+1:nT,:);
              decv(1000*j+1:nT,:)= Kernel(obj.kernelType, xnewk',train.patterns',param.k)*model.Beta+ ones(size(xnewk,1),1)*model.b;
          else
              decv =  Kernel(obj.kernelType, test.patterns',train.patterns',param.k)*model.Beta+ ones(nT,1)*model.b;
          end
            pred(decv<- param.threshold) = -1; pred(decv> param.threshold) = 1;
            nclass =  train.nOfClasses;
            expLosses=zeros(size(pred,1),nclass);
            for i=1:nclass,
                expLosses(:,i) = sum(pred == repmat(model.Code(i,:),size(pred,1),1),2);
            end
            [~,pred] = max(expLosses,[],2);
            pred = model.labelSet(pred);
                                  
        end
        
        
        
    end
end
