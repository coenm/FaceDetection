using System.Threading.Tasks;
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
        public static async Task Main(string[] args)
        {
            await using var exiftoolService = new AsyncExifToolRotationService();
            var identification = new DLibFaceIdentification(exiftoolService);

            var algorithms = new List<IFaceDetection>
            {
                new OpenCvDnnCaffe(),
                new OpenCvDnnTensorflow(),
                new DLibHog(exiftoolService),
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

            /*
            foreach (var img in Directory.GetFiles(inputDir, "*.jpg", SearchOption.TopDirectoryOnly))
            {
                foreach (var algo in algorithms)
                {
                    var faceCount = await algo.ProcessAsync(img, outputDir);
                    System.Console.WriteLine($"{algo.Name} found {faceCount} faces");
                }
            }
            */

            await identification.ProcessAsync(Directory.GetFiles(inputDir, "*.jpg", SearchOption.TopDirectoryOnly));

            foreach (var algo in algorithms)
            {
                if (algo is IDisposable disposable)
                    disposable.Dispose();
            }

            if (identification is IAsyncDisposable adis)
                await adis.DisposeAsync();

            if (identification is IDisposable dis)
                dis.Dispose();

            System.Console.WriteLine("Done");
            System.Console.ReadLine();
        }
    }
}
