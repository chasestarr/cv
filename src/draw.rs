use crate::image::Image;
use crate::point::Point;

pub fn mark(im: &mut Image, p: &Point) {
    let x = p.x as i32;
    let y = p.y as i32;
    for j in -9..10 {
        im.set_pixel(x + j, y, 0, 1.0);
        im.set_pixel(x, y + j, 0, 1.0);
        im.set_pixel(x + j, y, 1, 0.0);
        im.set_pixel(x, y + j, 1, 0.0);
        im.set_pixel(x + j, y, 2, 1.0);
        im.set_pixel(x, y + j, 2, 1.0);
    }
}

fn draw_line_low(im: &mut Image, start: &Point, end: &Point) {
    let dx = end.x - start.x;
    let mut dy = end.y - start.y;
    let mut yi = 1;
    if dy < 0.0 {
        yi = -1;
        dy = -dy;
    }
    let mut p = 2.0 * dy - dx;
    let mut y = start.y as i32;
    for x in start.x as i32..end.x as i32 {
        im.set_pixel(x, y, 0, 0.5);
        im.set_pixel(x, y, 1, 1.0);
        im.set_pixel(x, y, 2, 0.0);

        if p > 0.0 {
            y = y + yi;
            p = p - 2.0 * dx;
        }
        p = p + 2.0 * dy;
    }
}

fn draw_line_high(im: &mut Image, start: &Point, end: &Point) {
    let mut dx = end.x - start.x;
    let dy = end.y - start.y;
    let mut xi = 1;
    if dx < 0.0 {
        xi = -1;
        dx = -dx;
    }
    let mut p = 2.0 * dx - dy;
    let mut x = start.x as i32;
    for y in start.y as i32..end.y as i32 {
        im.set_pixel(x, y, 0, 0.5);
        im.set_pixel(x, y, 1, 1.0);
        im.set_pixel(x, y, 2, 0.0);

        if p > 0.0 {
            x = x + xi;
            p = p - 2.0 * dy;
        }
        p = p + 2.0 * dx;
    }
}

// TODO(chase): make line color arg
pub fn draw_line(im: &mut Image, start: &Point, end: &Point) {
    if (end.y - start.y).abs() < (end.x - start.x).abs() {
        if start.x > end.x {
            draw_line_low(im, end, start);
        } else {
            draw_line_low(im, start, end);
        }
    } else {
        if start.y > end.y {
            draw_line_high(im, end, start);
        } else {
            draw_line_high(im, start, end);
        }
    }
}

// pub fn draw_line(im: &mut Image, start: &Point, end: &Point) {
//     let mut x1 = start.x as i32;
//     let mut y1 = start.y as i32;
//     let mut x2 = end.x as i32;
//     let mut y2 = end.y as i32;
//     if (y2 - y1).abs() > (x2 - x1).abs() {
//         swap(&mut x1, &mut y1);
//         swap(&mut x2, &mut y2);
//     }
//     if x1 > x2 {
//         swap(&mut x1, &mut x2);
//         swap(&mut y1, &mut y2);
//     }

//     let dx = x2 - x1;
//     let dy = y2 - y1;
//     let mut p = 2 * dy - dx;
//     let mut y = y1;
//     for x in x1..x2 {
//         im.set_pixel(x, y, 0, 1.0);
//         im.set_pixel(x, y, 1, 0.0);
//         im.set_pixel(x, y, 2, 1.0);

//         if p > 0 {
//             y = y + 1;
//             p = p + 2 * dy;
//         }
//         p = p + 2 * dy;
//     }
// }
