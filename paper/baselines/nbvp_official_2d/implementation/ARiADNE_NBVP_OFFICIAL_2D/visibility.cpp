// Planar occupancy-grid geometry for the RH-NBVP port.
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
// Grid traversal from real-valued cell-center coordinates. At a corner,
// inspect both adjacent cells as well as the diagonal cell.
bool clear(const uint8_t* map, int width, int height, double x0, double y0,
           double x1, double y1, bool require_free) {
  int x = static_cast<int>(std::floor(x0 + 0.5));
  int y = static_cast<int>(std::floor(y0 + 0.5));
  const int end_x = static_cast<int>(std::floor(x1 + 0.5));
  const int end_y = static_cast<int>(std::floor(y1 + 0.5));
  const double dx = x1 - x0, dy = y1 - y0;
  const int sx = dx < 0 ? -1 : 1, sy = dy < 0 ? -1 : 1;
  const auto valid = [&](int col, int row) {
    if (col < 0 || col >= width || row < 0 || row >= height) return false;
    const uint8_t cell = map[row * width + col];
    return require_free ? cell == 255 : cell != 1;
  };
  while (valid(x, y)) {
    if (x == end_x && y == end_y) return true;
    const double tx = x == end_x ? INFINITY : (x + sx * 0.5 - x0) / dx;
    const double ty = y == end_y ? INFINITY : (y + sy * 0.5 - y0) / dy;
    if (std::abs(tx - ty) < 1e-12) {
      if (!valid(x + sx, y) || !valid(x, y + sy)) return false;
      x += sx;
      y += sy;
    } else if (tx < ty) {
      x += sx;
    } else {
      y += sy;
    }
  }
  return false;
}
}  // namespace

extern "C" int edge_free(const uint8_t* map, int width, int height,
                         double x0, double y0, double x1, double y1) {
  return clear(map, width, height, x0, y0, x1, y1, true);
}

extern "C" int visible_unknown(const uint8_t* map, int width, int height,
                               double x, double y, double radius) {
  const int left = std::max(0, static_cast<int>(std::ceil(x - radius)));
  const int right = std::min(width - 1, static_cast<int>(std::floor(x + radius)));
  const int top = std::max(0, static_cast<int>(std::ceil(y - radius)));
  const int bottom = std::min(height - 1, static_cast<int>(std::floor(y + radius)));
  int count = 0;
  for (int row = top; row <= bottom; ++row) {
    for (int col = left; col <= right; ++col) {
      if (map[row * width + col] != 127) continue;
      const double dx = col - x, dy = row - y;
      if (dx * dx + dy * dy <= radius * radius &&
          clear(map, width, height, x, y, col, row, false)) ++count;
    }
  }
  return count;
}
