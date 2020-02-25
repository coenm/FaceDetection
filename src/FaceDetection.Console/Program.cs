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
                var baseFilename = new FileInfo(img).Name + ".output.jpg";
                // foreach (var algo in algorithms)
                {
                    var outputFile = Path.Combine(outputDir, baseFilename);
                    File.Copy(img, outputFile);

                    var faces1 = await algorithms[0].ProcessAsync(img);
                    faces1 = faces1.Where(x => x.Confidence > 0.5);

                    var faces2 = await algorithms[1].ProcessAsync(img);
                    faces2 = faces2.Where(x => x.Confidence > 0.5);

                    var faces3 = await algorithms[2].ProcessAsync(img);
                    faces3 = faces3.Where(x => x.Confidence == null || x.Confidence > 0.5);

                    var f1 = faces1.Select(x => x.Position).ToList();
                    var f2 = faces2.Select(x => x.Position).ToList();
                    var f3 = faces3.Select(x => x.Position).ToList();
                    foreach (var face in ClusterFacePositions.ClusterFacesByPosition(f1, f2, f3))
                    {
                        Rectangle? union = null;

                        if (face.Set1Index.HasValue)
                        {
                            if (union == null)
                                union = f1[face.Set1Index.Value];
                            union = Rectangle.Union(union.Value, f1[face.Set1Index.Value]);
                        }

                        if (face.Set2Index.HasValue)
                        {
                            if (union == null)
                                union = f2[face.Set2Index.Value];
                            union = Rectangle.Union(union.Value, f2[face.Set2Index.Value]);
                        }

                        if (face.Set3Index.HasValue)
                        {
                            if (union == null)
                                union = f3[face.Set3Index.Value];
                            union = Rectangle.Union(union.Value, f3[face.Set3Index.Value]);
                        }

                        if (union.HasValue)
                            faceDrawer.Draw(outputFile, outputFile, Color.Aqua, new Face[]{ new Face { Position = union.Value }});
                    }

                    if (faces1.Any())
                        faceDrawer.Draw(outputFile, outputFile, Color.Red, faces1);

                    if (faces2.Any())
                        faceDrawer.Draw(outputFile, outputFile, Color.Blue, faces2);

                    if (faces3.Any())
                        faceDrawer.Draw(outputFile, outputFile, Color.Green, faces3);
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
