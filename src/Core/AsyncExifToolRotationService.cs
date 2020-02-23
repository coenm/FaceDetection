using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Threading.Tasks;
using CoenM.ExifToolLib;

namespace Core
{
    public class AsyncExifToolRotationService : IImageRotationService, IAsyncDisposable
    {
        private AsyncExifTool asyncExifTool;

        public AsyncExifToolRotationService()
        {
            var configuration = new AsyncExifToolConfiguration("exiftool.exe", Encoding.UTF8, Environment.NewLine, new List<string>());
            asyncExifTool = new AsyncExifTool(configuration);
            asyncExifTool.Initialize();
        }

        public async Task<int> GetImageRotationDegreesAsync(string filename)
        {
            var result = await asyncExifTool.ExecuteAsync(new[] { "-Orientation", "-n", filename });
            result = result.ToLower();
            result = result.Replace("orientation", string.Empty);
            result = result.Replace(":", string.Empty);
            result = result.Trim();

            if (!int.TryParse(result, NumberStyles.None, new NumberFormatInfo(), out var intResult))
            {
                return 0;
            }

            return intResult switch
            {
                1 => 0,
                3 => 180,
                6 => 90,
                8 => 270,
                _ => 0
            };
        }

        public async ValueTask DisposeAsync()
        {
            await asyncExifTool.DisposeAsync();
        }
    }
}