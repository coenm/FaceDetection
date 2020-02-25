using Core;

namespace FaceDetection.DLibDotNet
{
    public class DLibRectangleExtractor
    {
        private readonly IImageRotationService imageRotationService;

        public DLibRectangleExtractor(IImageRotationService imageRotationService)
        {
            this.imageRotationService = imageRotationService;
        }

        /*
        public async Task ExtractAsync(string filename, System.Drawing.Rectangle[] faces)
        {
            using var img = await DlibHelpers.LoadRotatedImage(imageRotationService, filename);

            foreach (var face in faces)
            {
                var dPoint = new[]
                    {
                        new DPoint(face.Left, face.Top),
                        new DPoint(face.Right, face.Top),
                        new DPoint(face.Left, face.Bottom),
                        new DPoint(face.Right, face.Bottom),
                    };

                var width = face.Width;
                var height = face.Height;

                var img2 = Dlib.ExtractImage4Points(img, dPoint, width, height);

                var b = img2.ToBytes();

                var img3 = Dlib.LoadImageData<RgbPixel>(b, (uint) height, (uint) width, (uint) (b.Length / height));
            }
        }
    */
    }
}