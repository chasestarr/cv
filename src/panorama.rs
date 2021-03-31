use rand::{thread_rng, Rng};

use crate::draw::{draw_line, mark};
use crate::harris::{corner_detector, Descriptor};
use crate::image::{bilinear_interpolate, copy_image, make_image, Image};
use crate::matrix::{make_matrix, matrix_invert, matrix_mult_matrix, solve_system, Matrix};
use crate::point::Point;

// A match between two points in an image.
// point p, q: x,y coordinates of the two matching pixels.
// int ai, bi: indexes in the descriptor array. For eliminating duplicates.
// float distance: the distance between the descriptors for the points.
#[derive(Clone)]
pub struct Match {
    p: Point,
    q: Point,
    ai: usize,
    bi: usize,
    distance: f32,
}

fn match_compare(a: &Match, b: &Match) -> std::cmp::Ordering {
    return a.distance.partial_cmp(&b.distance).unwrap();
}

fn both_images(a: &Image, b: &Image) -> Image {
    let mut im = make_image(
        a.width + b.width,
        a.height.max(b.height),
        a.channels.max(b.channels),
    );

    for c in 0..a.channels {
        for y in 0..a.height as i32 {
            for x in 0..a.width as i32 {
                im.set_pixel(x, y, c, a.get_pixel(x, y, c));
            }
        }
    }

    for c in 0..b.channels {
        for y in 0..b.height as i32 {
            for x in 0..b.width as i32 {
                im.set_pixel(x + a.width as i32, y, c, b.get_pixel(x, y, c));
            }
        }
    }

    return im;
}

fn draw_matches(a: &Image, b: &Image, matches: &Vec<Match>, inliers: usize) -> Image {
    let mut both = both_images(a, b);

    for i in 0..matches.len() {
        let bx = matches[i].p.x as i32;
        let ex = matches[i].q.x as i32;
        let by = matches[i].p.y as i32;
        let ey = matches[i].q.y as i32;
        draw_line(
            &mut both,
            &Point {
                x: bx as f32,
                y: by as f32,
            },
            &Point {
                x: ex as f32 + a.width as f32,
                y: ey as f32,
            },
        );
    }

    return both;
}

pub fn find_and_draw_matches(a: &Image, b: &Image, sigma: f32, thresh: f32, nms: u8) -> Image {
    let ad = corner_detector(a, sigma, thresh, nms);
    let bd = corner_detector(b, sigma, thresh, nms);
    let matches = match_descriptors(&ad, &bd);

    let mut mark_a = copy_image(a);
    let mut mark_b = copy_image(b);
    for i in 0..ad.len() {
        mark(&mut mark_a, &ad[i].p);
    }
    for i in 0..bd.len() {
        mark(&mut mark_b, &bd[i].p);
    }
    return draw_matches(&mark_a, &mark_b, &matches, 0);
}

fn project_point(h: &Matrix, p: &Point) -> Point {
    let n = Matrix {
        rows: 3,
        cols: 1,
        data: vec![vec![p.x as f64], vec![p.y as f64], vec![1.0]],
    };
    let r = matrix_mult_matrix(h, &n);

    let x = r.data[0][0];
    let y = r.data[1][0];
    let w = r.data[2][0];

    return Point {
        x: (x / w) as f32,
        y: (y / w) as f32,
    };
}

fn l1_distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    return sum;
}

fn match_descriptors(a: &Vec<Descriptor>, b: &Vec<Descriptor>) -> Vec<Match> {
    let mut matches = vec![];
    for i in 0..a.len() {
        let mut min = std::f32::MAX;
        let mut b_idx = 0;
        for j in 0..b.len() {
            let dist = l1_distance(&a[i].data, &b[j].data);
            if dist < min {
                min = dist;
                b_idx = j;
            }
        }
        matches.push(Match {
            ai: i,
            bi: b_idx,
            p: a[i].p.clone(),
            q: b[b_idx].p.clone(),
            distance: 0.0,
        });
    }

    matches.sort_by(|a, b| a.bi.cmp(&b.bi));
    matches.dedup_by(|a, b| a.bi == b.bi);

    return matches;
}

fn point_distance(p: &Point, q: &Point) -> f32 {
    let a = (p.x - q.x).abs();
    let b = (p.y - q.y).abs();
    return (a.powi(2) + b.powi(2)).sqrt();
}

fn is_match_inlier(h: &Matrix, m: &Match, thresh: f32) -> bool {
    let a = project_point(h, &m.p);
    let b = project_point(h, &m.q);
    return point_distance(&a, &b) < thresh;
}

fn randomize_matches(matches: &mut Vec<Match>) {
    let mut rng = thread_rng();
    for i in 0..matches.len() - 2 {
        let j = rng.gen_range(i..matches.len());
        let tmp = matches[i].clone();
        matches[i] = matches[j].clone();
        matches[j] = tmp;
    }
}

fn compute_homography(matches: &Vec<Match>) -> Option<Matrix> {
    let mut m = make_matrix(matches.len() * 2, 8);
    let mut b = make_matrix(matches.len() * 2, 1);
    for i in 0..matches.len() {
        let x = matches[i].p.x as f64;
        let xp = matches[i].q.x as f64;
        let y = matches[i].p.y as f64;
        let yp = matches[i].p.y as f64;
        m.data[i * 2] = vec![x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * yp];
        m.data[i * 2 + 1] = vec![0.0, 0.0, 0.0, x, y, 1.0, -x * xp, -y * yp];
        b.data[i * 2][0] = xp;
        b.data[i * 2 + 1][0] = yp;
    }
    let a_opt = solve_system(&m, &b);
    match a_opt {
        Some(a) => {
            let mut h = make_matrix(3, 3);
            for i in 0..8 {
                h.data[i / 3][i % 3] = a.data[i][0];
            }
            h.data[2][2] = 1.0;
            return Some(h);
        }
        None => {
            return None;
        }
    }
}

pub fn ransac(matches: &mut Vec<Match>, thresh: f32, k: usize, cutoff: usize) -> Option<Matrix> {
    let mut best_inlier_count = 0;
    let mut best_homography = None;
    for i in 0..k {
        randomize_matches(matches);
        if let Some(h) = compute_homography(&matches[0..4].to_vec()) {
            let mut inlier_count = 0;
            for j in 0..matches.len() {
                if is_match_inlier(&h, &matches[j], thresh) {
                    inlier_count += 1;
                }
            }

            if inlier_count >= cutoff {
                return Some(h);
            }

            if inlier_count > best_inlier_count {
                best_inlier_count = inlier_count;
                best_homography = Some(h);
            }
        }
    }
    return best_homography;
}

fn combine_images(a: &Image, b: &Image, h: &Matrix) -> Option<Image> {
    if let Some(h_inv) = matrix_invert(&h) {
        let c1 = project_point(&h_inv, &Point { x: 0.0, y: 0.0 });
        let c2 = project_point(
            &h_inv,
            &Point {
                x: (b.width - 1) as f32,
                y: 0.0,
            },
        );
        let c3 = project_point(
            &h_inv,
            &Point {
                x: 0.0,
                y: (b.height - 1) as f32,
            },
        );
        let c4 = project_point(
            &h_inv,
            &Point {
                x: (b.width - 1) as f32,
                y: (b.height - 1) as f32,
            },
        );

        let bot_right = Point {
            x: c1.x.max(c2.x).max(c3.x).max(c4.x),
            y: c1.y.max(c2.y).max(c3.y).max(c4.y),
        };
        let top_left = Point {
            x: c1.x.min(c2.x).min(c3.x).min(c4.x),
            y: c1.y.min(c2.y).min(c3.y).min(c4.y),
        };

        let dx: f32 = (0.0 as f32).min(top_left.x);
        let dy: f32 = (0.0 as f32).min(top_left.y);
        let width = ((a.width as f32).max(bot_right.x) - dx) as u32;
        let height = ((a.height as f32).max(bot_right.y) - dy) as u32;

        if width > 7000 || height > 7000 {
            println!("miscalculated homography");
            return None;
        }

        let mut im = make_image(width, height, a.channels);
        for c in 0..a.channels {
            for y in 0..a.height as i32 {
                for x in 0..a.width as i32 {
                    let val = a.get_pixel(x, y, c);
                    im.set_pixel(x + dx as i32, y + dy as i32, c, val);
                }
            }
        }

        for c in 0..a.channels {
            for y in (top_left.y as i32)..(bot_right.y as i32) {
                for x in (top_left.x as i32)..(bot_right.x as i32) {
                    let p = project_point(
                        h,
                        &Point {
                            x: x as f32,
                            y: y as f32,
                        },
                    );
                    if p.x >= 0.0 && p.x < b.width as f32 && p.y >= 0.0 && p.y < b.height as f32 {
                        im.set_pixel(
                            x - dx as i32,
                            y - dy as i32,
                            c,
                            bilinear_interpolate(b, p.x, p.y, c),
                        );
                    }
                }
            }
        }

        return Some(im);
    }

    return None;
}

// Create a panoramam between two images.
// image a, b: images to stitch together.
// float sigma: gaussian for harris corner detector. Typical: 2
// float thresh: threshold for corner/no corner. Typical: 1-5
// int nms: window to perform nms on. Typical: 3
// float inlier_thresh: threshold for RANSAC inliers. Typical: 2-5
// int iters: number of RANSAC iterations. Typical: 1,000-50,000
// int cutoff: RANSAC inlier cutoff. Typical: 10-100
pub fn panorama_image(
    a: &Image,
    b: &Image,
    sigma: f32,
    thresh: f32,
    nms: u8,
    inlier_thresh: f32,
    iters: usize,
    cutoff: usize,
) -> Option<Image> {
    let ad = corner_detector(a, sigma, thresh, nms);
    let bd = corner_detector(b, sigma, thresh, nms);
    let mut matches = match_descriptors(&ad, &bd);
    if let Some(h) = ransac(&mut matches, inlier_thresh, iters, cutoff) {
        return combine_images(a, b, &h);
    }
    return None;
}
