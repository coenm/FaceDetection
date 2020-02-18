using System.Collections.Generic;
using Core;
using FaceDetection.OpenCv;

namespace FaceDetection.Console
{
    class Program
    {
        static void Main(string[] args)
        {

            var algorithms = new List<IFaceDetection>
            {
                new OpenCvDnnCaffe(),
                new OpenCvDnnTensorflow(),
            };

            var img = "liverpool.jpg";

            foreach (var algo in algorithms)
            {
                var faceCount = algo.Process(img, "samples\\");
                System.Console.WriteLine($"{algo.Name} found {faceCount} faces");
            }

            System.Console.WriteLine("Done");
            System.Console.ReadLine();
        }
    }
}
