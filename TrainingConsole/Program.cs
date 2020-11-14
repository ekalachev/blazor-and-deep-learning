using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using SchemaLibrary;

namespace TrainingConsole
{
    static class Program
    {
        static void Main(string[] args)
        {
            var dataPath = Path.Combine(Environment.CurrentDirectory, "iris.data");
            
            // 1. Initialize MLContext
            var mlContext = new MLContext();

            // 2. Load the data
            var data = mlContext.Data.LoadFromTextFile<ModelInput>(dataPath, separatorChar: ',');

            // 3. Shuffle the data
            var shuffledData = mlContext.Data.ShuffleRows(data);

            // 4. Define the data preparation and training pipeline.
            var pipeline =
                mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                    .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes())
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // 5. Train with cross-validation
            var cvResults = mlContext.MulticlassClassification.CrossValidate(shuffledData, pipeline);

            // 6. Get the highest performing model and its accuracy
            var model =
                cvResults
                    .OrderByDescending(fold => fold.Metrics.MacroAccuracy)
                    .Select(fold => (fold.Model, fold.Metrics.MacroAccuracy))
                    .First();

            Console.WriteLine($"Top performing model's macro-accuracy: {model.MacroAccuracy}");

            // 7. Save the model
            // TODO upload the model to AZURE Storage
            mlContext.Model.Save(model.Model, data.Schema, "model.zip");

            Console.WriteLine("Model trained");
        }
    }
}