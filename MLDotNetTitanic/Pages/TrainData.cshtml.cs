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
    public class TrainDataModel : PageModel
    {
        private readonly IHostingEnvironment _env;
        public List<TrainModel> Trains { get; private set; }
        public string EvaluateQualityMessage { get; private set; }
        public double TrainingTime { get; private set; }

        public TrainDataModel(IHostingEnvironment env)
        {
            _env = env;
        }

        public void OnGet()
        {
            this.BindData();
        }

        public void OnPost()
        {
            this.BindData();
            this.MachineLearning();
        }

        private void BindData()
        {
            var filePath = _env.ContentRootFileProvider.GetFileInfo("Data/train.csv").PhysicalPath;
            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                Trains = csv.GetRecords<TrainModel>().ToList();
            }
        }

        private void MachineLearning()
        {
            var startTime = DateTime.Now;
            var filePath = Path.Combine(_env.ContentRootPath, "Data");
            var data = Trains.Select(t => new ModelInput
            {
                PassengerId = t.PassengerId,
                Survived = t.Survived,
                Pclass = t.Pclass,
                Name = t.Name,
                Sex = t.Sex,
                Age = t.Age.GetValueOrDefault(),
                SibSp = t.SibSp,
                Parch = t.Parch,
                Ticket = t.Ticket,
                Fare = t.Fare.GetValueOrDefault(),
                Cabin = t.Cabin,
                Embarked = t.Embarked
            });
            EvaluateQualityMessage = ModelBuilder.CreateModel(data, filePath);
            var endTime = DateTime.Now;

            TrainingTime = (endTime - startTime).TotalSeconds;
        }
    }
}