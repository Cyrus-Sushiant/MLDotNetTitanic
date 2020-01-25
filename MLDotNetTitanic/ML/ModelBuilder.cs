using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using MLDotNetTitanic.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MLDotNetTitanic.ML
{
    public class ModelBuilder
    {
        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private static MLContext mlContext = new MLContext(seed: 1);

        public static string CreateModel(IEnumerable<ModelInput> data, string modelFilePath)
        {
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromEnumerable<ModelInput>(data);

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            // Evaluate quality of Model
            string evaluateQuality = Evaluate(mlContext, trainingDataView, trainingPipeline);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Save model
            SaveModel(mlContext, mlModel, modelFilePath, trainingDataView.Schema);

            return evaluateQuality;
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("Sex", "Sex"), new InputOutputColumnPair("Embarked", "Embarked") })
                                      .Append(mlContext.Transforms.Categorical.OneHotHashEncoding(new[] { new InputOutputColumnPair("Cabin", "Cabin") }))
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "Sex", "Embarked", "Cabin", "Pclass", "Age", "SibSp", "Parch" }));

            // Set the training algorithm 
            var trainer = mlContext.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options() { NumberOfIterations = 150, LearningRate = 0.05196403f, NumberOfLeaves = 57, MinimumExampleCountPerLeaf = 50, UseCategoricalSplit = false, HandleMissingValue = false, MinimumExampleCountPerGroup = 50, MaximumCategoricalSplitPointCount = 16, CategoricalSmoothing = 20, L2CategoricalRegularization = 0.1, Booster = new GradientBooster.Options() { L2Regularization = 1, L1Regularization = 0 }, LabelColumnName = "Survived", FeatureColumnName = "Features" });
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            ITransformer model = trainingPipeline.Fit(trainingDataView);
            return model;
        }

        private static string Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Survived");
            return PrintBinaryClassificationFoldsAverageMetrics(crossValidationResults);
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelPath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            //mlContext.Model.Save(mlModel, modelInputSchema, modelPath);

            var path = Path.Combine(modelPath, "MLModel.zip");
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
            {
                mlContext.Model.Save(mlModel, modelInputSchema, fs);
                fs.Close();
            }
        }

        public static string PrintBinaryClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);
            var AccuracyAverage = AccuracyValues.Average();
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);

            return $"Average Accuracy:    {AccuracyAverage:0.###}  - Standard deviation: ({AccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({AccuraciesConfidenceInterval95:#.###})";
        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }
    }
}
