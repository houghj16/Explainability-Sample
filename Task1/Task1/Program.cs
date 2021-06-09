using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Task1
{
    class Program
    {

        /// <summary>
        /// model input class for TaxiFarePred.
        /// </summary>
        #region model input class
        public class ModelInput
        {
            [ColumnName(@"vendor_id"), LoadColumn(0)]
            public string Vendor_id { get; set; }

            [ColumnName(@"rate_code"), LoadColumn(1)]
            public float Rate_code { get; set; }

            [ColumnName(@"passenger_count"), LoadColumn(2)]
            public float Passenger_count { get; set; }

            [ColumnName(@"trip_time_in_secs"), LoadColumn(3)]
            public float Trip_time_in_secs { get; set; }

            [ColumnName(@"trip_distance"), LoadColumn(4)]
            public float Trip_distance { get; set; }

            [ColumnName(@"payment_type"), LoadColumn(5)]
            public string Payment_type { get; set; }

            [ColumnName(@"fare_amount"), LoadColumn(6)]
            public float Fare_amount { get; set; }
        }

        #endregion

        /// <summary>
        /// model output class for TaxiFarePred.
        /// </summary>
        #region model output class
        public class ModelOutput
        {
            public float Score { get; set; }
        }
        #endregion

        private static string TrainDataPath = "Data/taxi-fare-train.csv";
        private static string TestDataPath = "Data/taxi-fare-test.csv";
        private static string ModelPath = "TaxiFareModel.zip";

        static void Main(string[] args)
        {
            // Create the MLContext and load the data
            MLContext mlContext = new MLContext();
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(TrainDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(TestDataPath, hasHeader: true, separatorChar: ',');

            // Create, train, evaluate and save a model
            TransformerChain<ITransformer> trainedModel = BuildTrainEvaluateAndSaveModel(mlContext, trainingDataView, testDataView);

            // Make a single test prediction loading the model from .ZIP file
            TestSinglePrediction(mlContext);

            // Calculate the Permuation Feature Importance (PFI)
            CalculatePermutationFeatureImportance(mlContext, trainingDataView, trainedModel);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        private static TransformerChain<ITransformer> BuildTrainEvaluateAndSaveModel(MLContext mlContext, IDataView trainingDataView, IDataView testDataView)
        {
            // Run AutoML regression experiment
            Console.WriteLine("=============== Training the model ===============");
            Console.WriteLine($"Running AutoML regression experiment for 120 seconds...");
            ExperimentResult<RegressionMetrics> experimentResult = mlContext.Auto()
                .CreateRegressionExperiment(120)
                .Execute(trainingDataView, labelColumnName: @"fare_amount");

            // STEP 3: Evaluate the model and print metrics
            Console.WriteLine("===== Evaluating model's accuracy with test data =====");
            RunDetail<RegressionMetrics> best = experimentResult.BestRun;
            ITransformer trainedModel = best.Model;
            TransformerChain<ITransformer> trainedModelChain = (TransformerChain<ITransformer>)best.Model;
            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: @"fare_amount", scoreColumnName: "Score");

            // Print metrics from top model
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {best.TrainerName} regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");

            // STEP 4: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModelChain;
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            Console.WriteLine("=============== Testing prediction engine ===============");

            // Sample: 
            // vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            // VTS,1,1,1140,3.75,CRD,15.5

            var taxiTripSample = new ModelInput()
            {
                Vendor_id = "VTS",
                Rate_code = 1,
                Passenger_count = 1,
                Trip_time_in_secs = 1140,
                Trip_distance = 3.75f,
                Payment_type = "CRD",
                Fare_amount = 0 // To predict. Actual/Observed = 15.5
            };

            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            // Score
            var predictedResult = predEngine.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {predictedResult.Score:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void CalculatePermutationFeatureImportance(MLContext mlContext, IDataView trainingDataView, TransformerChain<ITransformer> trainedModel)
        {

        }
    }
}
