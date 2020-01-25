using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLDotNetTitanic.ML
{
    public class ModelInput
    {
        public float PassengerId { get; set; }
        public bool Survived { get; set; }
        public float Pclass { get; set; }
        public string Name { get; set; }
        public string Sex { get; set; }
        public float Age { get; set; }
        public float SibSp { get; set; }
        public float Parch { get; set; }
        public string Ticket { get; set; }
        public float Fare { get; set; }
        public string Cabin { get; set; }
        public string Embarked { get; set; }
    }
}
