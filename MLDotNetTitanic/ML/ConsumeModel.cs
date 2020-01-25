using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MLDotNetTitanic.ML
{
    public class ConsumeModel
    {
        // For more info on consuming ML.NET models, visit https://aka.ms/model-builder-consume
        // Method for consuming model in your app
        public static ModelOutput Predict(ModelInput input, string rootPath)
        {

            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Load model & create prediction engine
            string modelPath = Path.Combine(rootPath, "Data", "MLModel.zip");
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use model to make prediction on input data
            ModelOutput result = predEngine.Predict(input);
            return result;
        }

        public static IEnumerable<bool> Predict(IEnumerable<ModelInput> data, string rootPath)
        {

            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Load Trained Model
            string modelPath = Path.Combine(rootPath, "Data", "MLModel.zip");
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            // Predicted Data
            IDataView inputData = mlContext.Data.LoadFromEnumerable<ModelInput>(data);
            IDataView predictions = mlModel.Transform(inputData);

            // Get Predictions
            var result = predictions.GetColumn<bool>("PredictedLabel");

            return result;
        }
    }
}
