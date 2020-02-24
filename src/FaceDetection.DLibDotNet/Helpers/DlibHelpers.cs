using System;
using System.Threading.Tasks;
using Core;
using DlibDotNet;

namespace FaceDetection.DLibDotNet.Helpers
{
    internal static class DlibHelpers
    {
        public static async Task<Array2D<RgbPixel>> LoadRotatedImage(IImageRotationService imageRotationService, string filename)
        {
            var result = await imageRotationService.GetImageRotationDegreesAsync(filename);

            var radiansRotation = result switch
            {
                0 => 0,
                180 => Math.PI,
                90 => 4.71238898038469D, // 1.5 * Math.PI
                270 => 1.5707963267949D, // 0.5 * Math.PI
                _ => 0
            };

            var origImg = Dlib.LoadImage<RgbPixel>(filename);

            if (radiansRotation == 0d)
                return origImg;

            // load the image
            var img = new Array2D<RgbPixel>();
            Dlib.RotateImage(origImg, img, radiansRotation, InterpolationTypes.Bilinear);

            origImg.Dispose();
            return img;
        }
    }
}