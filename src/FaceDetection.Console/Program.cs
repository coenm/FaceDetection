using System.ComponentModel.DataAnnotations;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using Core.Persistence;
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

            var faceDrawer = new OpenCvDrawFaces();

            var algorithms = new List<IFaceDetection>
            {
                new OpenCvDnnCaffe(),
                new OpenCvDnnTensorflow(),
                new DLibHog(exiftoolService),
                // new DLibDnnMmod(),
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
                {
                    var outputFile = img + ".output.jpg";
                    File.Copy(img, outputFile);

                    var faces1 = await algorithms[0].ProcessAsync(img);
                    faces1 = faces1.Where(x => x.Confidence > 0.5);
                    if (faces1.Any())
                        faceDrawer.Draw(outputFile, outputFile, Color.Aqua, faces1);

                    var faces2 = await algorithms[1].ProcessAsync(img);
                    faces2 = faces2.Where(x => x.Confidence > 0.5);
                    if (faces2.Any())
                        faceDrawer.Draw(outputFile, outputFile, Color.BlueViolet, faces2);

                    var faces3 = await algorithms[2].ProcessAsync(img);
                    faces3 = faces3.Where(x => x.Confidence == null || x.Confidence > 0.5);
                    if (faces3.Any())
                        faceDrawer.Draw(outputFile, outputFile, Color.Chartreuse, faces3);

                    List<Rectangle> f1 = faces1.Select(x => x.Position).ToList();
                    List<Rectangle> f2 = faces2.Select(x => x.Position).ToList();
                    List<Rectangle> f3 = faces3.Select(x => x.Position).ToList();
                    var result = ClusterFacePositions.Abc(f1, f2, f3);

                }
            }

            // await identification.ProcessAsync(Directory.GetFiles(inputDir, "*.jpg", SearchOption.TopDirectoryOnly));

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
