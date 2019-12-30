%% SECTION 1 (Set random seed)
% Set to 5 to be able to reproduce results
rng(5);

%% SECTION 2 (Import data from file)
% Read MS Excel worksheet data to table
CreditCardDataTable = readtable('Default-of-Credit-Card-Clients.xls', ...
    'Sheet', 'Data', 'Range', 'A3:Y30002', 'ReadVariableNames', false);

% Remove 'ID' column (first column) as this is not needed
CreditCardDataTable = removevars(CreditCardDataTable, {'Var1'});

% Rename variables (features and class)
CreditCardDataTable.Properties.VariableNames = {'LimitBalance', 'Gender', ...
    'Education', 'MaritalStatus', 'Age', 'PaymentSep', 'PaymentAug', ...
    'PaymentJul', 'PaymentJun', 'PaymentMay', 'PaymentApr', 'BillAmountSep', ...
    'BillAmountAug', 'BillAmountJul', 'BillAmountJun', 'BillAmountMay', ...
    'BillAmountApr', 'PaymentAmountSep', 'PaymentAmountAug', 'PaymentAmountJul', ...
    'PaymentAmountJun', 'PaymentAmountMay', 'PaymentAmountApr', 'Class'};

clearvars ans;

%% SECTION 3 (Check for missing values)
% Total amount of 'NaN' (missing) values in 'CreditCardDataTable'
MissingValues = sum(sum(ismissing(CreditCardDataTable)));

if MissingValues == 0
    fprintf('\n');
    fprintf('%d missing values in data table. \n', MissingValues);
    fprintf('\n');
else
    fprintf('\n');
    fprintf('%d missing values in data table. \n', MissingValues);
    fprintf('\n');
end

clearvars MissingValues ans;

%% SECTION 4 (Group unidentified groups in applicable features to 'Other group)
% 'Education' feature (group 0, 5, 6 to 4 ('Other'))
for I = [0, 5, 6]
    EducationIndex = find(CreditCardDataTable.(3) == I);
    CreditCardDataTable.Education(EducationIndex) = 4;
end

% 'MaritalStatus' feature (group 0 to 3 ('Other'))
MaritalStatusIndex = find(CreditCardDataTable.(4) == 0);
CreditCardDataTable.MaritalStatus(MaritalStatusIndex) = 3;

clearvars I EducationIndex MaritalStatusIndex ans;

%% SECTION 5 (Exploratory data analysis of 'CreditCardDataTable')
% Display class distribution
% Note: 0 = 'No Default' and 1 = 'Default'
fprintf('Distribution of classes in data table: \n');
tabulate(table2array(CreditCardDataTable(:, 24)));
fprintf('\n');

% Class distribution: 
% 0 = 77.88%% (23364 instances)
% 1 = 22.12% (6636 instances)

% Create table only containing numerical features for statistical analysis
NumericalCreditCardDataTable = removevars(CreditCardDataTable, {'Gender', ...
    'Education', 'MaritalStatus', 'PaymentSep', 'PaymentAug', 'PaymentJul', ...
    'PaymentJun', 'PaymentMay', 'PaymentApr', 'Class'});

% Calcualte mean and standard deviation of numerical features
MSDNumericalFeatures = zeros(14, 2);
for I = 1:14
    MSDNumericalFeatures(I, 1) = mean(NumericalCreditCardDataTable.(I));
    MSDNumericalFeatures(I, 2) = std(NumericalCreditCardDataTable.(I));
end

% Store mean and standard deviation of numerical features in table
MSDNumericalFeaturesTable = array2table(MSDNumericalFeatures, 'VariableNames', ...
    {'Mean', 'StandardDeviation'}, 'RowNames', {'LimitBalance', 'Age', ...
    'BillAmountSep', 'BillAmountAug', 'BillAmountJul', 'BillAmountJun', ...
    'BillAmountMay', 'BillAmountApr', 'PaymentAmountSep', ...
    'PaymentAmountAug', 'PaymentAmountJul', 'PaymentAmountJun', ...
    'PaymentAmountMay', 'PaymentAmountApr'});

% Display table
fprintf('Numerical features mean and standard deviation table: \n');
disp(MSDNumericalFeaturesTable);

% Display 'LimitBalance' distribution histograms
figure;
% Data with both classes ('No Default' and 'Default')
subplot(3, 1, 1);
histogram(CreditCardDataTable.(1), 50, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('(Credit) Limit Balance');
ylabel('Frequency');
title('Limit Balance Distribution');
% 'No Default' data
subplot(3, 1, 2);
histogram(CreditCardDataTable.LimitBalance(CreditCardDataTable.(24) == 0), ...
    50, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('(Credit) Limit Balance');
ylabel('Frequency');
title('Limit Balance (No Default) Distribution');
% 'Default' data
subplot(3, 1, 3);
histogram(CreditCardDataTable.LimitBalance(CreditCardDataTable.(24) == 1), ...
    50, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('(Credit) Limit Balance');
ylabel('Frequency');
title('Limit Balance (Default) Distribution');

% Display 'Gender' vs 'LimitBalance' boxplots
figure;
% Data with both classes ('No Default' and 'Default')
subplot(1, 3, 1);
boxplot(CreditCardDataTable.LimitBalance, CreditCardDataTable.Gender, ...
    'Symbol', 'ro', 'Colors', 'k');
% Set boxplot subplot properties
BoxplotProperties = gca;
BoxplotProperties.FontSize = 8;
% Set face colour
Object = findobj(gca, 'Tag', 'Box');
for I = 1:length(Object)
    patch(get(Object(I), 'XData'), get(Object(I), 'YData'), 'r', 'FaceAlpha', ...
        0.5);
end
xlabel('Gender');
xticklabels({'Male', 'Female'});
ylabel('(Credit) Limit Balance')
title('Boxplots - Gender vs Limit Balance');
% 'No Default' data
subplot(1, 3, 2);
boxplot(CreditCardDataTable.LimitBalance(CreditCardDataTable.(24) == 0), ...
    CreditCardDataTable.Gender(CreditCardDataTable.(24) == 0), ...
    'Symbol', 'ro', 'Colors', 'k');
% Set boxplot subplot properties
BoxplotProperties = gca;
BoxplotProperties.FontSize = 8;
% Set face colour
Object = findobj(gca, 'Tag', 'Box');
for I = 1:length(Object)
    patch(get(Object(I), 'XData'), get(Object(I), 'YData'), 'r', 'FaceAlpha', ...
        0.5);
end
xlabel('Gender');
xticklabels({'Male', 'Female'});
ylabel('(Credit) Limit Balance')
title('Boxplots (No Default) - Gender vs Limit Balance');
% 'Default' data
subplot(1, 3, 3);
boxplot(CreditCardDataTable.LimitBalance(CreditCardDataTable.(24) == 1), ...
    CreditCardDataTable.Gender(CreditCardDataTable.(24) == 1), ...
    'Symbol', 'ro', 'Colors', 'k');
% Set boxplot subplot properties
BoxplotProperties = gca;
BoxplotProperties.FontSize = 8;
% Set face colour
Object = findobj(gca, 'Tag', 'Box');
for I = 1:length(Object)
    patch(get(Object(I), 'XData'), get(Object(I), 'YData'), 'r', 'FaceAlpha', ...
        0.5);
end
xlabel('Gender');
xticklabels({'Male', 'Female'});
ylabel('(Credit) Limit Balance')
title('Boxplots (Default) - Gender vs Limit Balance');

% Display 'Education' vs 'LimitBalance' boxplots
figure;
% Data with both classes ('No Default' and 'Default')
subplot(1, 3, 1);
boxplot(CreditCardDataTable.LimitBalance, CreditCardDataTable.Education, ...
    'Symbol', 'ro', 'Colors', 'k');
% Set boxplot subplot properties
BoxplotProperties = gca;
BoxplotProperties.FontSize = 8;
% Set face colour
Object = findobj(gca, 'Tag', 'Box');
for I = 1:length(Object)
    patch(get(Object(I), 'XData'), get(Object(I), 'YData'), 'r', 'FaceAlpha', ...
        0.5);
end
xlabel('Education');
xticklabels({'Graduate School', 'University', 'High School', 'Other'});
ylabel('(Credit) Limit Balance')
title('Boxplots - Education vs Limit Balance');
% 'No Default' data
subplot(1, 3, 2);
boxplot(CreditCardDataTable.LimitBalance(CreditCardDataTable.(24) == 0), ...
    CreditCardDataTable.Education(CreditCardDataTable.(24) == 0), ...
    'Symbol', 'ro', 'Colors', 'k');
% Set boxplot subplot properties
BoxplotProperties = gca;
BoxplotProperties.FontSize = 8;
% Set face colour
Object = findobj(gca, 'Tag', 'Box');
for I = 1:length(Object)
    patch(get(Object(I), 'XData'), get(Object(I), 'YData'), 'r', 'FaceAlpha', ...
        0.5);
end
xlabel('Education');
xticklabels({'Graduate School', 'University', 'High School', 'Other'});
ylabel('(Credit) Limit Balance')
title('Boxplots (No Default) - Education vs Limit Balance');
% 'Default' data
subplot(1, 3, 3);
boxplot(CreditCardDataTable.LimitBalance(CreditCardDataTable.(24) == 1), ...
    CreditCardDataTable.Education(CreditCardDataTable.(24) == 1), ...
    'Symbol', 'ro', 'Colors', 'k');
% Set boxplot subplot properties
BoxplotProperties = gca;
BoxplotProperties.FontSize = 8;
% Set face colour
Object = findobj(gca, 'Tag', 'Box');
for I = 1:length(Object)
    patch(get(Object(I), 'XData'), get(Object(I), 'YData'), 'r', 'FaceAlpha', ...
        0.5);
end
xlabel('Education');
xticklabels({'Graduate School', 'University', 'High School', 'Other'});
ylabel('(Credit) Limit Balance')
title('Boxplots (Default) - Education vs Limit Balance');

% Display pearson's correlation coefficient of each feature on heatmap
% Define features of which are to be contained in heatmap
CorrelationFeatures = {'LimitBalance', 'Education', 'Gender', 'MaritalStatus', ...
    'Age', 'PaymentSep', 'PaymentAug', 'PaymentJul', 'PaymentJun', ...
    'PaymentMay', 'PaymentApr', 'BillAmountSep', ...
    'BillAmountAug', 'BillAmountJul', 'BillAmountJun', 'BillAmountMay', ...
    'BillAmountApr', 'PaymentAmountSep', 'PaymentAmountAug', 'PaymentAmountJul', ...
    'PaymentAmountJun', 'PaymentAmountMay', 'PaymentAmountApr'};
PCC = corr(CreditCardDataTable{:, CorrelationFeatures}, 'Type', 'pearson');
% Display 'PCC' on heatmap
figure;
PCCHeatMap = heatmap(PCC, 'colormap', colormap('gray'));
HeatMapProperties = gca;
xlabel('Feature');
HeatMapProperties.XData = ({'LimitBalance', 'Education', 'Gender', 'MaritalStatus', ...
    'Age', 'PaymentSep', 'PaymentAug', 'PaymentJul', 'PaymentJun', ...
    'PaymentMay', 'PaymentApr', 'BillAmountSep', ...
    'BillAmountAug', 'BillAmountJul', 'BillAmountJun', 'BillAmountMay', ...
    'BillAmountApr', 'PaymentAmountSep', 'PaymentAmountAug', 'PaymentAmountJul', ...
    'PaymentAmountJun', 'PaymentAmountMay', 'PaymentAmountApr'});
ylabel('Feature');
HeatMapProperties.YData = ({'LimitBalance', 'Education', 'Gender', 'MaritalStatus', ...
    'Age', 'PaymentSep', 'PaymentAug', 'PaymentJul', 'PaymentJun', ...
    'PaymentMay', 'PaymentApr', 'BillAmountSep', ...
    'BillAmountAug', 'BillAmountJul', 'BillAmountJun', 'BillAmountMay', ...
    'BillAmountApr', 'PaymentAmountSep', 'PaymentAmountAug', 'PaymentAmountJul', ...
    'PaymentAmountJun', 'PaymentAmountMay', 'PaymentAmountApr'});
title('Pearson Correlation Coefficient Heatmap of Features');

% Complete all figures before moving onto next section of code
drawnow;

clearvars NumericalCreditCardDataTable MSDNumericalFeatures I BoxplotProperties ...
    Object CorrelationFeatures PCC PCCHeatMap HeatMapProperties ans;

%% SECTION 6 (Resample dataset)
% Undersample majority class ('No Default') to balance class distribution
% Set random seed again (for precaution)
rng(5);

% Undersampling
% Extract 6636 samples of 'No Default' and 'Default' classes 
% Note: 'Default' class contains 6636 samples
NoDefaultIndices = find(CreditCardDataTable.Class == 0);
NDPermutation = randperm(size(NoDefaultIndices, 1));
NoDefaultIndexSamples = NoDefaultIndices(NDPermutation(1:6636), :);
DefaultIndexSamples = find(CreditCardDataTable.Class == 1);

% Combine undersampled 'No Default' indices and total 'Default' indices
SampledIndices = [NoDefaultIndexSamples, DefaultIndexSamples];
% Shuffle
SampledCreditCardDataTable = table2array(CreditCardDataTable(SampledIndices, ...
    :));
SampledCreditCardDataTable = SampledCreditCardDataTable(randperm(size(...
    SampledCreditCardDataTable, 1)), :);
SampledCreditCardDataTable = array2table(SampledCreditCardDataTable);

% Display class distribution of 'SampledCreditCardDataTable'
fprintf('Distribution of classes in data table: \n');
tabulate(table2array(SampledCreditCardDataTable(:, 24)));
fprintf('\n');

clearvars NoDefaultIndices NDPermutation NoDefaultIndexSamples DefaultIndexSamples ...
    SampledIndices ans;

%% SECTION 7 (Split sampled dataset into training set and test set)
% Training set and test set allocation: 75% : 25%
% Split 'SampledCreditCardDataTable' into features and class
% Note: normalize feature data using z-score (X - Mu) / (Sigma)
Features = normalize(SampledCreditCardDataTable(:, 1:23));
Class = SampledCreditCardDataTable(:, 24);

Split = height(SampledCreditCardDataTable) * 0.75;

% Split 'Features' and 'Class' into training set and test set
TrainingFeatures = table2array(Features(1:Split, :));
TestFeatures = table2array(Features((Split + 1):end, :));
TrainingClass = table2array(Class(1:Split, :));
TestClass = table2array(Class((Split + 1):end, :));

% Change class labels from '0 and 1' to '1 and 2' for NN toolbox
TrainingClass(TrainingClass == 1) = 2;
TrainingClass(TrainingClass == 0) = 1;
TestClass(TestClass == 1) = 2; 
TestClass(TestClass == 0) = 1;

% Display class distribution in training set and test set
fprintf('Training set class distribution: \n')
tabulate(TrainingClass);
fprintf('\n');
fprintf('Test set class distribution \n');
tabulate(TestClass);
fprintf('\n');

clearvars Features Class Split ans;

%%-------------------------------------------------------------------------
%%-------------------------MULTILAYER PERCEPTRON---------------------------
%%-------------------------------------------------------------------------

%% SECTION 8 (Implement grid search to find optimal parameters for MLP)
% Set random seed again (for precaution)
rng(5);

% Convert training set features and class tables into arrays
MLPTrainingFeatures = transpose(TrainingFeatures);
MLPTrainingClass = full(ind2vec(transpose(TrainingClass)));

% Count variable
Count = 1;

% Maximum number of epochs
Epochs = 500;

% Define array to store results of grid search
MLPGSResults = zeros(625, 6);

% Grid search with 10-fold cross validation
% Time grid search
tic 
for NoHiddenNeurons = [10, 20, 30, 40, 50]
    for NoHiddenLayers = [1, 2, 3, 4, 5]
        for LearningRate = [0.05, 0.1, 0.3, 0.6, 0.9]
            for Momentum = [0.1, 0.3, 0.5, 0.7, 0.9]
                % Train network on 'gradient descent with momentum and adaptive learning rate backpropagation'
                Net = feedforwardnet(repelem(NoHiddenNeurons, NoHiddenLayers), 'traingdx');
                % Specify maximum epochs for network training
                Net.trainParam.epochs = Epochs;
                % Specify learning rate for network training
                Net.trainParam.lr = LearningRate;
                % Specify error goal for network training 
                Net.trainParam.goal = 0.001;
                % Specify momentum for network training
                Net.trainParam.mc = Momentum;
                % 10 fold cross validation (generate training set indices)
                CVIndices = crossvalind('Kfold', MLPTrainingFeatures(1, :), 10);
                % Define array to store error of each fold
                CVPerformance = zeros(1, 10);
                % 10 fold cross validation loop
                for I = 1:10
                    % Check which samples are in in the ith fold
                    TestIndex = (CVIndices == I);
                    TrainIndex = ~TestIndex;
                    GetTrainIndex = find(TrainIndex);
                    GetTestIndex = find(TestIndex);
                    % Divide samples up using 'indices' divide function
                    Net.divideFcn = 'divideind';
                    % Retrieve training sample indices 
                    Net.divideParam.trainInd = GetTrainIndex;
                    % Retrieve testing sample indices
                    Net.divideParam.testInd = GetTestIndex;
                    
                    % Train MLP network on specified architecture
                    [TrainedNetwork, Train] = train(Net, MLPTrainingFeatures, ...
                        MLPTrainingClass);
                    
                    % Fit trained MLP network on training data for performance evaluation
                    MLPPredictedCVModel = TrainedNetwork(MLPTrainingFeatures);
                    Difference = gsubtract(MLPTrainingClass, MLPPredictedCVModel);
                    CVPerformance(:, I) = crossentropy(TrainedNetwork, MLPTrainingClass, ...
                        MLPPredictedCVModel);
                    % Note: 'crossentropy' has been used for a performance metric as this is a classification task
                end 
                % Display results of selected hyperparamters
                fprintf('Epochs = %d \n', Epochs);
                fprintf('Number of Hidden Neurons = %d \n', NoHiddenNeurons);
                fprintf('Number of Hidden Layers = %d \n', NoHiddenLayers);
                fprintf('Learning Rate = %0.2f \n', LearningRate);
                fprintf('Momentum = %0.2f \n', Momentum);
                fprintf('Average 10-Fold CV Crossentropy Result = %0.5f \n', mean(CVPerformance));
                fprintf('\n');
                % Store results in 'MLPGSResults' array for analysis        
                MLPGSResults(Count, 1) = Epochs;
                MLPGSResults(Count, 2) = NoHiddenNeurons;
                MLPGSResults(Count, 3) = NoHiddenLayers;
                MLPGSResults(Count, 4) = LearningRate;
                MLPGSResults(Count, 5) = Momentum;
                MLPGSResults(Count, 6) = mean(CVPerformance);
                Count = Count + 1;
            end
        end
    end
end
MLPGSTime = toc;

% Results of grid search (optimal hyperparameters for MLP)
% Epochs = 500 
% Number of Hidden Neurons = 10 
% Number of Hidden Layers = 3 
% Learning Rate = 0.60 
% Momentum = 0.90 
% Average 10-Fold CV Crossentropy Result = 0.19883 

% Grid search time
% 52264.5704 seconds

clearvars MLPTrainingFeatures MLPTrainingClass Count Epochs NoHiddenNeurons ...
    NoHiddenLayers LearningRate Momentum Net CVIndices CVPerformance I TestIndex ...
    TrainIndex GetTrainIndex GetTestIndex TrainedNetwork Train MLPPredictedCVModel ...
    Difference CVPerformance ans;

%% SECTION 9 (Re-train MLP on optimal hyperparameters)
% Set random seed again (for precaution)
rng(5);

% Convert training set features and class tables into arrays
MLPTrainingFeatures = transpose(TrainingFeatures);
MLPTrainingClass = full(ind2vec(transpose(TrainingClass)));

% Maximum number of epochs
Epochs = 500;

% Store optimal hyperparameter values in variables
NoHiddenNeurons = 10;
NoHiddenLayers = 3;
LearningRate = 0.3;
Momentum = 0.9;

% Train network on 'gradient descent with momentum and adaptive learning rate backpropagation'
Net = feedforwardnet(repelem(NoHiddenNeurons, NoHiddenLayers), 'traingdx');
% Specify maximum epochs for network training
Net.trainParam.epochs = Epochs;
% Specify learning rate for network training
Net.trainParam.lr = LearningRate;
% Specify error goal for network training 
Net.trainParam.goal = 0.001;
% Specify momentum for network training
Net.trainParam.mc = Momentum;
% 10 fold cross validation (generate training set indices)
CVIndices = crossvalind('Kfold', MLPTrainingFeatures(1, :), 10);
% Define array to store error of each fold
CVPerformance = zeros(1, 10);
% Define cell array to store MLP network prediction arrays from 10 fold cross validation
PredictedCVStorage = cell(1, 10, 1);
% 10 fold cross validation loop
for I = 1:10
    % Check which samples are in in the ith fold
    TestIndex = (CVIndices == I);
    TrainIndex = ~TestIndex;
    GetTrainIndex = find(TrainIndex);
    GetTestIndex = find(TestIndex);
    % Divide samples up using 'indices' divide function
    Net.divideFcn = 'divideind';
    % Retrieve training sample indices 
    Net.divideParam.trainInd = GetTrainIndex;
    % Retrieve testing sample indices
    Net.divideParam.testInd = GetTestIndex;
                    
    % Train MLP network on specified architecture
    [TrainedNetwork, Train] = train(Net, MLPTrainingFeatures, MLPTrainingClass);
                    
    % Fit trained MLP network on training data for performance evaluation
    MLPPredictedCVModel = TrainedNetwork(MLPTrainingFeatures);
    % Store ith fold prediction array into 'PredictedCVStorage' cell array
    PredictedCVStorage{I} = MLPPredictedCVModel;
    Difference = gsubtract(MLPTrainingClass, MLPPredictedCVModel);
    CVPerformance(:, I) = crossentropy(TrainedNetwork, MLPTrainingClass, ...
        MLPPredictedCVModel);
end 

% Display results of selected hyperparamters
fprintf('Epochs = %d \n', Epochs);
fprintf('Number of Hidden Neurons = %d \n', NoHiddenNeurons);
fprintf('Number of Hidden Layers = %d \n', NoHiddenLayers);
fprintf('Learning Rate = %0.2f \n', LearningRate);
fprintf('Momentum = %0.2f \n', Momentum);
fprintf('Average 10-Fold CV Crossentropy Result = %0.5f \n', mean(CVPerformance));
fprintf('\n');

% Encode 10-fold prediction arrays with binary values to assess performance
for I = 1:size(PredictedCVStorage, 2)
    for J = 1:size(PredictedCVStorage{I}, 2)
        if PredictedCVStorage{I}(1, J) > PredictedCVStorage{I}(2, J)
            PredictedCVStorage{I}(1, J) = 1;
            PredictedCVStorage{I}(2, J) = 0;
        else
            PredictedCVStorage{I}(1, J) = 0;
            PredictedCVStorage{I}(2, J) = 1;
        end
    end
end

% Calculate accuracy, recall, precision and F1-score of trained MLP network through confusion matrix
TNA = zeros(1, 10);
TPA = zeros(1, 10);
FNA = zeros(1, 10);
FPA = zeros(1, 10);
for I = 1:10
    MLPCM = confusionmat(vec2ind(MLPTrainingClass), vec2ind(PredictedCVStorage{I}));
    TNA(I) = MLPCM(1, 1);
    TPA(I) = MLPCM(2, 2);
    FNA(I) = MLPCM(2, 1);
    FPA(I) = MLPCM(1, 2);
end
% Calculate mean of the 10 confusion matrices to evaluate overall performance
TN = round(mean(TNA));
TP = round(mean(TPA));
FN = round(mean(FNA));
FP = round(mean(FPA));

% Plot confusion chart with TN TP FN FP values
figure;
confusionchart([TN FP; FN TP], [0, 1], ...
   'DiagonalColor', 'k', 'OffDiagonalColor', 'w');
title('Multilayer Perceptron Training Confusion Matrix');

% Define performance metrics 
% Training accuracy
MLPTrainingAccuracy = (TP + TN) / (TP + TN + FP + FN); 
% Training recall
MLPTrainingRecall = TP / (TP + FN);
% Training precision
MLPTrainingPrecision = TP / (TP + FP);
% Training F1-score
MLPTrainingFS = (2 * (MLPTrainingRecall * MLPTrainingPrecision)) / (MLPTrainingRecall + MLPTrainingPrecision);

% Display MLP network training accuracy, recall, precision and F1 score
fprintf('MLP training accuracy = %0.5f \n', MLPTrainingAccuracy);
fprintf('MLP training recall = %0.5f \n', MLPTrainingRecall);
fprintf('MLP training precision = %0.5f \n', MLPTrainingPrecision);
fprintf('MLP training F1 score = %0.5f \n', MLPTrainingFS);
fprintf('\n');

clearvars Epochs NoHiddenNeurons NoHiddenLayers LearningRate Momentum Net ...
    CVIndices I TestIndex TrainIndex GetTrainIndex GetTestIndex MLPPredictedCVModel ...
    Difference J MLPCM TNA TPA FNA FPA TN TP FN FP MLPTrainingAccuracy ...
    MLPTrainingRecall MLPTrainingPrecision MLPTrainingFS ans;

%% SECTION 10 (Test MLP network)
% Set random seed again (for precaution)
rng(5);

% Convert test set features and class tables into arrays
MLPTestFeatures = transpose(TestFeatures);
MLPTestClass = full(ind2vec(transpose(TestClass)));

% Fit trained MLP network to test features
MLPPredictedTestModel = TrainedNetwork(MLPTestFeatures);
% Define probabilities for ROC
MLPPredictedTestModelROC = MLPPredictedTestModel;

% Evaluate 'crossentropy' performance of 'MLPPredictedTestModel'
MLPCE = crossentropy(TrainedNetwork, MLPTestClass, MLPPredictedTestModel);
fprintf('MLP Test Model - Crossentropy Result = %0.5f \n', MLPCE);
fprintf('\n');

% Encode test prediction array with binary values to assess performance
for I = 1:size(MLPPredictedTestModel, 2)
    if MLPPredictedTestModel(1, I) > MLPPredictedTestModel(2, I)
        MLPPredictedTestModel(1, I) = 1;
        MLPPredictedTestModel(2, I) = 0;
    else
        MLPPredictedTestModel(1, I) = 0;
        MLPPredictedTestModel(2, I) = 1;
    end
end

% Calculate accuracy, recall, precision and F1-score of trained MLP network on test set through confusion matrix
MLPCM = confusionmat(vec2ind(MLPTestClass), vec2ind(MLPPredictedTestModel));
TN = MLPCM(1, 1);
TP = MLPCM(2, 2);
FN = MLPCM(2, 1);
FP = MLPCM(1, 2);

% Plot confusion chart with TN TP FN FP values
figure;
confusionchart([TN FP; FN TP], [0, 1], ...
   'DiagonalColor', 'k', 'OffDiagonalColor', 'w');
title('Multilayer Perceptron Testing Confusion Matrix');

% Define performance metrics 
% Testing accuracy
MLPTestingAccuracy = (TP + TN) / (TP + TN + FP + FN); 
% Testing recall
MLPTestingRecall = TP / (TP + FN);
% Testing precision
MLPTestingPrecision = TP / (TP + FP);
% Testing F1-score
MLPTestingFS = (2 * (MLPTestingRecall * MLPTestingPrecision)) / (MLPTestingRecall + MLPTestingPrecision);

% Display MLP network training accuracy, recall, precision and F1 score
fprintf('MLP testing accuracy = %0.5f \n', MLPTestingAccuracy);
fprintf('MLP testing recall = %0.5f \n', MLPTestingRecall);
fprintf('MLP testing precision = %0.5f \n', MLPTestingPrecision);
fprintf('MLP testing F1 score = %0.5f \n', MLPTestingFS);
fprintf('\n');

clearvars MLPCE I MLPCM TN TP FN FP MLPTestingAccuracy MLPTestingRecall ...
    MLPTestingPrecision MLPTestingFS ans;

%%-------------------------------------------------------------------------
%%-------------------------SUPPORT VECTOR MACHINE--------------------------
%%-------------------------------------------------------------------------

%% SECTION 11 (Implement grid search to find optimal parameters for SVM)
% Set random seed again (for precaution)
rng(5);

% Create random 10-fold cross validation partition of the training class
SVMCV = cvpartition(TrainingClass, 'Kfold', 10);

% Grid search with 10-fold cross validation
% Time grid search
tic
% Loop through various SVM kernel functions
for SVMKernelFunction = {'rbf', 'linear', 'polynomial'}
    % If the kernel function is polynomial loop through different polynomial orders
    if strcmp(SVMKernelFunction{1}, 'polynomial')
        for PolynomialOrder = [2, 3, 4, 5, 6]
            TrainedSVM = fitcsvm(TrainingFeatures, TrainingClass, 'KernelFunction', ...
            SVMKernelFunction{1}, 'PolynomialOrder', PolynomialOrder, 'CVPartition', SVMCV);
            SVMLoss = kfoldLoss(TrainedSVM);
            fprintf('SVM | Kernel function = %s \n', SVMKernelFunction{1});
            fprintf('Polynomial Order %d \n', PolynomialOrder);
            fprintf('Average 10-Fold CV Loss Result = %0.5f \n', SVMLoss);
            fprintf('\n');
        end
    else
        TrainedSVM = fitcsvm(TrainingFeatures, TrainingClass, 'KernelFunction', ...
            SVMKernelFunction{1}, 'CVPartition', SVMCV);
        SVMLoss = kfoldLoss(TrainedSVM);
        fprintf('SVM | Kernel function = %s \n', SVMKernelFunction{1});
        fprintf('Average 10-Fold CV Loss Result = %0.5f \n', SVMLoss);
        fprintf('\n');
    end
end
SVMGSTime = toc;

% Results of grid search (optimal hyperparameters for SVM)
% SVM | Kernel function = polynomial 
% Polynomial Order 2 
% Average 10-Fold CV Loss Result = 0.30510 

% Grid search time
% 13096.86586 seconds

% Conduct random search for 'box constraint' and 'kernel scale' hyperparameters
% Define hyperparameter variable
SVMHP = hyperparameters('fitcsvm', TrainingFeatures, TrainingClass);
% Range for 'box constraint' 
SVMHP(2).Range = [0.001, 1];
% Range for 'kernel scale'
SVMHP(3).Range = [0.001, 1];

% Random search
% Time random search
tic
SVMHPRS = fitcsvm(TrainingFeatures, TrainingClass, 'KernelFunction', ...
    'polynomial', 'PolynomialOrder', 2, 'OptimizeHyperparameters', ...
    SVMHP, 'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Optimizer', 'randomsearch', 'MaxObjectiveEvaluations', 15, ...
    'CVPartition', SVMCV, 'ShowPlots', true));
SVMRSTime = toc;

% Results of random search
% |=======================================================================================|
% | Iter | Eval   | Objective   | Objective   | BestSoFar   | BoxConstraint|  KernelScale |
% |      | result |             | runtime     | (observed)  |              |              |
% |=======================================================================================|
% |    1 | Best   |     0.32319 |        2891 |     0.32319 |    0.0018345 |      0.29568 |
% |    2 | Accept |     0.49387 |      2396.3 |     0.32319 |       2.2097 |     0.082151 |
% |    3 | Accept |      0.4991 |      2623.3 |     0.32319 |    0.0016753 |    0.0051888 |
% |    4 | Accept |      0.4999 |      2666.2 |     0.32319 |       201.39 |    0.0019872 |
% |    5 | Accept |     0.49437 |      2454.4 |     0.32319 |     0.044864 |     0.023154 |
% |    6 | Accept |     0.46675 |      2367.3 |     0.32319 |       13.195 |     0.038779 |
% |    7 | Best   |     0.30932 |      2757.7 |     0.30932 |        2.535 |      0.99109 |
% |    8 | Best   |     0.30581 |      2931.5 |     0.30581 |      0.58763 |      0.87809 |
% |    9 | Accept |      0.4998 |      2559.8 |     0.30581 |       131.95 |    0.0013479 |
% |   10 | Accept |     0.45901 |      2143.5 |     0.30581 |      0.98697 |      0.36017 |
% |   11 | Accept |      0.4997 |      2644.3 |     0.30581 |    0.0021447 |    0.0017743 |
% |   12 | Accept |     0.48101 |        2375 |     0.30581 |    0.0015162 |    0.0090139 |
% |   13 | Accept |     0.48614 |      2342.7 |     0.30581 |     0.011329 |      0.03962 |
% |   14 | Accept |     0.46655 |      2109.9 |     0.30581 |       20.673 |      0.29324 |
% |   15 | Accept |     0.48744 |      2363.1 |     0.30581 |       875.32 |     0.057824 |
% 
% __________________________________________________________
% Optimization completed.
% MaxObjectiveEvaluations of 15 reached.
% Total function evaluations: 15
% Total elapsed time: 37658.2497 seconds.
% Total objective function evaluation time: 37625.9905
% 
% Best observed feasible point:
%     BoxConstraint    KernelScale
%     _____________    ___________
% 
%        0.58763         0.87809  
% 
% Observed objective function value = 0.30581
% Function evaluation time = 2931.506

% kfoldLoss = 0.4912 of optimal box constraint and kernel scale
% Default box constraint and kernel scale minimize kfoldLoss compared to random search

clearvars SVMCV SVMKernelFunction PolynomialOrder SVMLoss SVMHP SVMHPRS ans;

%% SECTION 12 (Re-train SVM on optimal hyperparameters)
% Set random seed again (for precaution)
rng(5);

% Create random 10-fold cross validation partition of the training class
SVMCV = cvpartition(TrainingClass, 'Kfold', 10);

% Train SVM model on optimal hyperparamters
TrainedSVM = fitcsvm(TrainingFeatures, TrainingClass, 'KernelFunction', ...
            'polynomial', 'PolynomialOrder', 2, 'CVPartition', SVMCV);

% Calculate 10-fold cross validation loss 
SVMTrainedLoss = kfoldLoss(TrainedSVM);

% Define array to store SVM training predictions from 10 fold cross validation
SVMTrainedPredictions = zeros(size(TrainingClass, 1), 10);

% Store 10-fold cross validation predictions
for I = 1:10
    SVMTrainedPredictions(:, I) = predict(TrainedSVM.Trained{I}, TrainingFeatures);
end

% Calculate accuracy, recall, precision and F1-score of trained SVM through confusion matrix
TNA = zeros(1, 10);
TPA = zeros(1, 10);
FNA = zeros(1, 10);
FPA = zeros(1, 10);
for I = 1:10
    SVMCM = confusionmat(TrainingClass, SVMTrainedPredictions(:, I));
    TNA(I) = SVMCM(1, 1);
    TPA(I) = SVMCM(2, 2);
    FNA(I) = SVMCM(2, 1);
    FPA(I) = SVMCM(1, 2);
end
% Calculate mean of the 10 confusion matrices to evaluate overall performance
TN = round(mean(TNA));
TP = round(mean(TPA));
FN = round(mean(FNA));
FP = round(mean(FPA));

% Plot confusion chart with TN TP FN FP values
figure;
confusionchart([TN FP; FN TP], [0, 1], ...
   'DiagonalColor', 'k', 'OffDiagonalColor', 'w');
title('Support Vector Machine Training Confusion Matrix');

% Define performance metrics 
% Training accuracy
SVMTrainingAccuracy = (TP + TN) / (TP + TN + FP + FN); 
% Training recall
SVMTrainingRecall = TP / (TP + FN);
% Training precision
SVMTrainingPrecision = TP / (TP + FP);
% Training F1-score
SVMTrainingFS = (2 * (SVMTrainingRecall * SVMTrainingPrecision)) / (SVMTrainingRecall + SVMTrainingPrecision);

% Display SVM training accuracy, recall, precision and F1 score
fprintf('SVM training accuracy = %0.5f \n', SVMTrainingAccuracy);
fprintf('SVM training recall = %0.5f \n', SVMTrainingRecall);
fprintf('SVM training precision = %0.5f \n', SVMTrainingPrecision);
fprintf('SVM training F1 score = %0.5f \n', SVMTrainingFS);
fprintf('\n');

clearvars I SVMCM TNA TPA FNA FPA TN TP FN FP SVMTrainingAccuracy ...
    SVMTrainingRecall SVMTrainingPrecision SVMTrainingFS ans;

%% SECTION 13 (Test SVM)
% Set random seed again (for precaution)
rng(5);

% Define array to store test predicitions
SVMPredictedTestModel = zeros(size(TestClass, 1), 10);

% Fit trained SVM to test features
for I = 1:10
    SVMPredictedTestModel(1:size(TestClass, 1), I) = predict(TrainedSVM.Trained{I}, TestFeatures);
end

% Define array to store test loss
SVMTestLoss = zeros(1, 10);

% Calculate average test loss on 10 trained models
for I = 1:10
    SVMTestLoss(1, I) = loss(TrainedSVM.Trained{I}, TestFeatures, TestClass);
end
% Use SVM model with minimum loss
[MinTestLoss, Index] = min(SVMTestLoss);
fprintf('SVM Test Model (%d) Test Loss = %0.5f \n', Index, MinTestLoss);
fprintf('\n');

% Define probabilities for ROC
[~, SVMScore] = predict(TrainedSVM.Trained{Index}, TestFeatures);

% Calculate accuracy, recall, precision and F1-score of trained SVM on test set through confusion matrix
SVMCM = confusionmat(TestClass, SVMPredictedTestModel(:, Index));
TN = SVMCM(1, 1);
TP = SVMCM(2, 2);
FN = SVMCM(2, 1);
FP = SVMCM(1, 2);

% Plot confusion chart with TN TP FN FP values
figure;
confusionchart([TN FP; FN TP], [0, 1], ...
   'DiagonalColor', 'k', 'OffDiagonalColor', 'w');
title('Support Vector Machine Testing Confusion Matrix');

% Define performance metrics 
% Testing accuracy
SVMTestingAccuracy = (TP + TN) / (TP + TN + FP + FN); 
% Testing recall
SVMTestingRecall = TP / (TP + FN);
% Testing precision
SVMTestingPrecision = TP / (TP + FP);
% Testing F1-score
SVMTestingFS = (2 * (SVMTestingRecall * SVMTestingPrecision)) / (SVMTestingRecall + SVMTestingPrecision);

% Display SVM training accuracy, recall, precision and F1 score
fprintf('SVM testing accuracy = %0.5f \n', SVMTestingAccuracy);
fprintf('SVM testing recall = %0.5f \n', SVMTestingRecall);
fprintf('SVM testing precision = %0.5f \n', SVMTestingPrecision);
fprintf('SVM testing F1 score = %0.5f \n', SVMTestingFS);
fprintf('\n');

clearvars I SVMTestLoss Index TN TP FN FP SVMTestingAccuracy ...
    SVMTestingRecall SVMTestingPrecision SVMTestingFS ans;

%% SECTION 14 (ROC for MLP and SVM test)
% Set random seed again (for precaution)
rng(5);

% Plot MLP Test ROC
MLPPredictedTestModelROC = transpose(MLPPredictedTestModelROC);
% No default class
[MLPX1, MLPY1, ~, MLPAUC1] = perfcurve(TestClass, MLPPredictedTestModelROC(:, 1), 1);
% Default class
[MLPX2, MLPY2, ~, MLPAUC2] = perfcurve(TestClass, MLPPredictedTestModelROC(:, 2), 2);
figure; 
hold on;
plot(MLPX1, MLPY1, 'r', 'Linewidth', 1.25);
plot(MLPX2, MLPY2, 'k', 'Linewidth', 1.25);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('MLP Test set ROC');
legend('0 = No Default', '1 = Default');
hold off;

% Display MLP AUC
fprintf('MLP Test ROC AUC - No Default %0.5f \n', MLPAUC1);
fprintf('MLP Test ROC AUC - Default %0.5f \n', MLPAUC2);
fprintf('\n');

% Plot SVM Test ROC
% No default class
[SVMX1, SVMY1, ~, SVMAUC1] = perfcurve(TestClass, SVMScore(:, 1), 1);
% Default class
[SVMX2, SVMY2, ~, SVMAUC2] = perfcurve(TestClass, SVMScore(:, 2), 2);
figure;
hold on;
plot(SVMX1, SVMY1, 'r', 'Linewidth', 1.25);
plot(SVMX2, SVMY2, 'k', 'Linewidth', 1.25);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('SVM Test set ROC');
legend('0 = No Default', '1 = Default');
hold off;

% Display SVM AUC
fprintf('SVM Test ROC AUC - No Default %0.5f \n', SVMAUC1);
fprintf('SVM Test ROC AUC - Default %0.5f \n', SVMAUC2);
fprintf('\n');

clearvars MLPX1 MLPY1 MLPAUC1 MLPX2 MLPY2 MLPAUC2 ...
    SVMX1 SVMY1 SVMAUC1 SVMX2 SVMY2 SVMAUC2