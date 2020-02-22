using System.Security.Cryptography.X509Certificates;
using FaceDetection.DLibDotNet;

namespace FaceDetection.Console
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Reflection;
    using Core;
    using OpenCv;

    public static class Program
    {
        public static void Main(string[] args)
        {

            var identification = new DLibFaceIdentification();

            var algorithms = new List<IFaceDetection>
            {
                new OpenCvDnnCaffe(),
                new OpenCvDnnTensorflow(),
                new DLibHog(),
            };

            var codebase = Assembly.GetEntryAssembly()?.GetName().CodeBase;
            if (codebase == null)
                return;
            var location = new Uri(codebase);

            var dir = new FileInfo(location.AbsolutePath).Directory;
            if (dir == null)
                return;

            var outputDir = Path.Combine(dir.FullName, "Output");
            if (Directory.Exists(outputDir) == false)
                Directory.CreateDirectory(outputDir);

            var inputDir = Path.Combine(dir.FullName, "images");
            if (!Directory.Exists(inputDir))
                return;

            foreach (var img in Directory.GetFiles(inputDir, "*.jpg", SearchOption.TopDirectoryOnly))
            {
                // foreach (var algo in algorithms)
                // {
                //     var faceCount = algo.Process(img, outputDir);
                //     System.Console.WriteLine($"{algo.Name} found {faceCount} faces");
                // }
            }


            identification.Process(Directory.GetFiles(inputDir, "*.jpg", SearchOption.TopDirectoryOnly));

            foreach (var algo in algorithms)
            {
                if (algo is IDisposable disposable)
                    disposable.Dispose();
            }

            if (identification is IDisposable dis)
                dis.Dispose();

            System.Console.WriteLine("Done");
            System.Console.ReadLine();
        }
    }
}
