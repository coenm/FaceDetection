using System.Drawing;

namespace Core.Persistence
{
    public class RectangleDto
    {
        public int Left { get; set; }
        public int Top { get; set; }
        public int Right { get; set; }
        public int Bottom { get; set; }

        public Rectangle ToRectangle()
        {
            return new Rectangle(Left, Top, Right-Left, Bottom-Top);
        }
    }
}