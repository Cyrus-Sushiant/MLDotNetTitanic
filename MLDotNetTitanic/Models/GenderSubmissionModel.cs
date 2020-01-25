using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLDotNetTitanic.Models
{
    public class GenderSubmissionModel
    {
        public int PassengerId { get; set; }
        public bool Survived { get; set; }
    }
}
