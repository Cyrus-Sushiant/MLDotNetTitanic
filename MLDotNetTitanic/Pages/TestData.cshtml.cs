using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CsvHelper;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.Hosting;
using MLDotNetTitanic.ML;
using MLDotNetTitanic.Models;

namespace MLDotNetTitanic
{
    public class TestDataModel : PageModel
    {
        private readonly IHostingEnvironment _env;

        [BindProperty(SupportsGet = true)]
        public int? PassengerId { get; set; }
        public List<ResultModel> ResultList { get; set; }
        public ResultModel ResultItem { get; set; }
        public int TotalItems => ResultList.Count;
        public int TotalItemsSubmission => ResultList.Count(i => i.ActualSurvived == i.PredictedSurvived);
        public double Accuracy => TotalItemsSubmission / (double)TotalItems;
        public TestDataModel(IHostingEnvironment env)
        {
            _env = env;
        }

        public void OnGet()
        {
            this.BindList();

            if (PassengerId.HasValue)
            {
                this.BindItem();
            }
        }

        private void BindList()
        {
            var testPath = _env.ContentRootFileProvider.GetFileInfo("Data/test.csv").PhysicalPath;
            var genderSubmissionPath = _env.ContentRootFileProvider.GetFileInfo("Data/gender_submission.csv").PhysicalPath;
            using (var readerTest = new StreamReader(testPath))
            using (var csvTest = new CsvReader(readerTest, CultureInfo.InvariantCulture))
            using (var readerGenderSubmission = new StreamReader(genderSubmissionPath))
            using (var csvGenderSubmission = new CsvReader(readerGenderSubmission, CultureInfo.InvariantCulture))
            {
                var genderSubmission = csvGenderSubmission.GetRecords<GenderSubmissionModel>();
                var result = csvTest.GetRecords<TestModel>().Select(r => new ResultModel
                {
                    PassengerId = r.PassengerId,
                    Pclass = r.Pclass,
                    Name = r.Name,
                    Sex = r.Sex,
                    Age = r.Age,
                    SibSp = r.SibSp,
                    Parch = r.Parch,
                    Ticket = r.Ticket,
                    Fare = r.Fare,
                    Cabin = r.Cabin,
                    Embarked = r.Embarked,
                    ActualSurvived = genderSubmission.First(gs => gs.PassengerId == r.PassengerId).Survived
                }).ToList();

                var dataPrediction = result.Select(r => new ModelInput
                {
                    PassengerId = r.PassengerId,
                    Survived = r.ActualSurvived,
                    Pclass = r.Pclass,
                    Name = r.Name,
                    Sex = r.Sex,
                    Age = r.Age.GetValueOrDefault(),
                    SibSp = r.SibSp,
                    Parch = r.Parch,
                    Ticket = r.Ticket,
                    Fare = r.Fare.GetValueOrDefault(),
                    Cabin = r.Cabin,
                    Embarked = r.Embarked
                });
                var predictionResult = ConsumeModel.Predict(dataPrediction, _env.ContentRootPath).ToList(); ;

                for (int i = 0; i < result.Count; i++)
                {
                    result[i].PredictedSurvived = predictionResult[i];
                }

                ResultList = result;
            }
        }

        private void BindItem()
        {
            ResultItem = ResultList.FirstOrDefault(i => i.PassengerId == PassengerId.Value);
            if (ResultItem != null)
            {
                var dataPrediction = new ModelInput
                {
                    PassengerId = ResultItem.PassengerId,
                    Survived = ResultItem.ActualSurvived,
                    Pclass = ResultItem.Pclass,
                    Name = ResultItem.Name,
                    Sex = ResultItem.Sex,
                    Age = ResultItem.Age.GetValueOrDefault(),
                    SibSp = ResultItem.SibSp,
                    Parch = ResultItem.Parch,
                    Ticket = ResultItem.Ticket,
                    Fare = ResultItem.Fare.GetValueOrDefault(),
                    Cabin = ResultItem.Cabin,
                    Embarked = ResultItem.Embarked
                };

                ResultItem.PredictedSurvived = ConsumeModel.Predict(dataPrediction, _env.ContentRootPath).Prediction;
            }
        }
    }
}