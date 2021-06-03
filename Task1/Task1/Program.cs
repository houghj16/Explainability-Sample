using System;
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
            [ColumnName(@"vendor_id")]
            [LoadColumn(0)]
            public string Vendor_id { get; set; }

            [ColumnName(@"rate_code")]
            [LoadColumn(1)]
            public float Rate_code { get; set; }

            [ColumnName(@"passenger_count")]
            [LoadColumn(2)]
            public float Passenger_count { get; set; }

            [ColumnName(@"trip_time_in_secs")]
            [LoadColumn(3)]
            public float Trip_time_in_secs { get; set; }

            [ColumnName(@"trip_distance")]
            [LoadColumn(4)]
            public float Trip_distance { get; set; }

            [ColumnName(@"payment_type")]
            [LoadColumn(5)]
            public string Payment_type { get; set; }

            [ColumnName(@"fare_amount")]
            [LoadColumn(6)]
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

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            MLContext mlContext = new MLContext();
            IDataView trainDataView = mlContext.Data.LoadFromTextFile<ModelInput>("my-data-file.csv", hasHeader: true);

            var experimentSettings = new RegressionExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 3600;
            experimentSettings.OptimizingMetric = RegressionMetric.MeanSquaredError;
            experimentSettings.CacheDirectory = null;

            RegressionExperiment experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);
            ExperimentResult<RegressionMetrics> experimentResult = experiment
                .Execute(trainDataView, @"fare_amount");

            RegressionMetrics metrics = experimentResult.BestRun.ValidationMetrics;
            Console.WriteLine($"R-Squared: {metrics.RSquared:0.##}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:0.##}");
        }
    }
}
