namespace FaceDetection.DLibDotNet
{
    using System.Threading.Tasks;
    using JetBrains.Annotations;

    using System;
    using DlibDotNet;
    using System.IO;
    using Core;

    public class DLibHog : IFaceDetection, IDisposable
    {
        private readonly IImageRotationService imageRotationService;
        private readonly FrontalFaceDetector detector;

        public DLibHog([NotNull] IImageRotationService imageRotationService)
        {
            this.imageRotationService = imageRotationService ?? throw new ArgumentNullException(nameof(imageRotationService));
            detector = Dlib.GetFrontalFaceDetector();
        }

        public string Name { get; } = "DLib-HOG-GetFrontalFaceDetector";

        public async Task<int> ProcessAsync(string inputFilename, string outputDirectory)
        {
            if (!File.Exists(inputFilename))
                throw new FileNotFoundException(nameof(inputFilename));

            using var img = await Helpers.LoadRotatedImage(imageRotationService, inputFilename);
            var faces = detector.Operator(img);

            foreach (var rect in faces)
            {
                Dlib.DrawRectangle(img, rect, new RgbPixel(0,255,0), 8U);
            }

            var origFilename = new FileInfo(inputFilename).Name;
            var outputFilename = Path.Combine(outputDirectory, $"{origFilename}_{Name}.jpg");

            Dlib.SaveJpeg(img, outputFilename, 25);

            return faces.Length;
        }

        public void Dispose()
        {
            detector.Dispose();
        }
    }
}
