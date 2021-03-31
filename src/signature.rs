use crate::image::{rgb_to_grayscale, Image};

type Position = (u32, u32);
type Window = (Position, Position);

// implemented from "An Image Signature for Any Kind of Image" at
// ../docs/an-image-signature-for-any-kind-of-image.pdf

// TODO exclude empty space from image edges
fn feature_bounds(im: &Image, _crop_percentiles: (f32, f32)) -> Window {
    return ((0, 0), (im.width, im.height));
}

fn compute_grid_points(size: u8, ((x_min, y_min), (x_max, y_max)): Window) -> Vec<Position> {
    let width = (x_max - x_min) as f32 / size as f32;
    let height = (y_max - y_min) as f32 / size as f32;

    let mut grid_points = vec![];
    for y in 0..size + 1 {
        for x in 0..size + 1 {
            grid_points.push((
                ((width * x as f32) + x_min as f32) as u32,
                ((height * y as f32) + y_min as f32) as u32,
            ))
        }
    }
    return grid_points;
}

fn sample_neighborhood(im: &Image, position: Position, w: u8) -> f32 {
    let mut sum = 0.0;

    let (x, y) = position;
    for yy in (y as i32 - w as i32)..((y + 1) as i32 + w as i32) {
        for xx in (x as i32 - w as i32)..((x + 1) as i32 + w as i32) {
            sum += im.get_pixel(xx, yy, 0);
        }
    }

    return sum / (w as f32 * 2.0 + 1.0);
}

fn position_index(x: u8, y: u8, size: u8) -> usize {
    return (y * size + x) as usize;
}

fn differentials(values: &Vec<f32>, size: u8, identical_tolerence: f32) -> Vec<i8> {
    let mut normalized = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let current = values[position_index(x, y, size)];
            let directions = vec![
                position_index(x - 1, y - 1, size),
                position_index(x, y - 1, size),
                position_index(x + 1, y - 1, size),
                position_index(x - 1, y, size),
                position_index(x + 1, y, size),
                position_index(x - 1, y + 1, size),
                position_index(x, y + 1, size),
                position_index(x + 1, y + 1, size),
            ];
            let dir_values: Vec<f32> = directions
                .iter()
                .map(|index| {
                    if *index < 0 || *index >= values.len() {
                        return current;
                    }
                    return values[*index];
                })
                .collect();

            let mut larger = Vec::new();
            let mut smaller = Vec::new();
            for v in dir_values.iter() {
                if *v > current + identical_tolerence {
                    larger.push(*v);
                } else if *v < current - identical_tolerence {
                    smaller.push(*v);
                }
            }
            let mut larger_sum = 0.0;
            for v in larger.iter() {
                larger_sum += *v;
            }
            let larger_avg = larger_sum / larger.len() as f32;
            let mut smaller_sum = 0.0;
            for v in smaller.iter() {
                smaller_sum += *v;
            }
            let smaller_avg = smaller_sum / smaller.len() as f32;

            for v in dir_values.iter() {
                if (current - v).abs() < identical_tolerence {
                    normalized.push(0);
                } else if *v < current {
                    if *v < smaller_avg {
                        normalized.push(-1);
                    } else {
                        normalized.push(-2);
                    }
                } else if *v > current {
                    if *v > larger_avg {
                        normalized.push(1);
                    } else {
                        normalized.push(2);
                    }
                }
            }
        }
    }

    return normalized;
}

pub fn sample_size(im: &Image) -> u8 {
    let size = (0.5 + im.width.min(im.height) as f32 / 20.0) as u8;
    return size.max(2);
}

pub fn signature(
    im: &Image,
    crop_percentiles: (f32, f32),
    grid_size: u8,
    sample_size: u8,
    identical_tolerence: f32,
) -> Vec<i8> {
    let bw = rgb_to_grayscale(im);
    let window = feature_bounds(&bw, crop_percentiles);
    let grid_points = compute_grid_points(grid_size, window);
    let mut samples = Vec::new();
    for i in 0..grid_points.len() {
        samples.push(sample_neighborhood(&bw, grid_points[i], sample_size));
    }
    return differentials(&samples, grid_size, identical_tolerence);
}

fn sub_vec(a: &Vec<i8>, b: &Vec<i8>) -> Vec<i8> {
    let mut c = Vec::new();
    for i in 0..a.len() {
        c.push(a[i] - b[i]);
    }
    return c;
}

fn norm_vec(v: &Vec<i8>) -> f32 {
    let mut sq_sum = 0.0;
    for i in 0..v.len() {
        sq_sum += v[i] as f32 * v[i] as f32;
    }
    return sq_sum.sqrt();
}

pub fn normalized_distance(a: &Vec<i8>, b: &Vec<i8>) -> f32 {
    let norm_diff = norm_vec(&sub_vec(a, b));
    let norm_a = norm_vec(a);
    let norm_b = norm_vec(b);
    return norm_diff / (norm_a + norm_b);
}
