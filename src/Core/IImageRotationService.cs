using System.Threading.Tasks;

namespace Core
{
    public interface IImageRotationService
    {
        Task<int> GetImageRotationDegreesAsync(string filename);
    }
}