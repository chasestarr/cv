use crate::image::{
  convolve_image, copy_image, make_gaussian_filter, make_gx_filter, make_gy_filter, make_image,
  Image,
};
use crate::point::Point;

// A descriptor for a point in an image.
// point p: x,y coordinates of the image pixel.
// float *data: the descriptor for the pixel.
pub struct Descriptor {
  pub p: Point,
  pub data: Vec<f32>,
}

fn structure_matrix(im: &Image, sigma: f32) -> Image {
  let x_sobel = convolve_image(&im, &make_gx_filter(), false);
  let y_sobel = convolve_image(&im, &make_gy_filter(), false);
  let mut s = make_image(im.width, im.height, 3);
  for y in 0..im.height as i32 {
    for x in 0..im.width as i32 {
      let ix = x_sobel.get_pixel(x, y, 0);
      let iy = y_sobel.get_pixel(x, y, 0);
      s.set_pixel(x, y, 0, ix * ix);
      s.set_pixel(x, y, 1, iy * iy);
      s.set_pixel(x, y, 2, ix * iy);
    }
  }
  let g = &make_gaussian_filter(sigma);
  return convolve_image(&s, g, true);
}

fn cornerness_response(im: &Image) -> Image {
  let alpha = 0.06;
  let mut r = make_image(im.width, im.height, 1);
  for y in 0..im.height as i32 {
    for x in 0..im.width as i32 {
      let xx = im.get_pixel(x, y, 0);
      let yy = im.get_pixel(x, y, 1);
      let xy = im.get_pixel(x, y, 2);
      let det = xx * yy - xy * xy;
      let trace = xx + yy;
      let v = det - alpha * trace * trace;
      r.set_pixel(x, y, 0, v);
    }
  }
  return r;
}

// non-max supression
fn nms_image(im: &Image, w: u8) -> Image {
  let mut r = copy_image(im);
  for y in 0..r.height as i32 {
    for x in 0..r.width as i32 {
      let current_response = r.get_pixel(x, y, 0);
      'neighborhood: for yy in (y - w as i32)..(y + 1 + w as i32) {
        for xx in (x - w as i32)..(x + 1 + w as i32) {
          let neighbor_response = r.get_pixel(xx, yy, 0);
          if neighbor_response > current_response {
            r.set_pixel(x, y, 0, -999999.0);
            break 'neighborhood;
          }
        }
      }
    }
  }
  return r;
}

fn describe_index(im: &Image, index: usize) -> Descriptor {
  let mut d = Descriptor {
    p: Point {
      x: index as f32 % im.width as f32,
      y: index as f32 / im.width as f32,
    },
    data: vec![],
  };

  let w = 5;
  for c in 0..im.channels {
    let cval = im.data[c as usize * (im.width * im.height) as usize + index];
    for x in -w / 2..(w + 1) / 2 {
      for y in -w / 2..(w + 1) / 2 {
        let val = im.get_pixel(x, y, c);
        d.data.push(cval - val);
      }
    }
  }

  return d;
}

pub fn corner_detector(im: &Image, sigma: f32, thresh: f32, nms: u8) -> Vec<Descriptor> {
  let s = structure_matrix(im, sigma);
  let r = cornerness_response(&s);
  let r_nms = nms_image(&r, nms);

  let mut descriptors = vec![];
  for i in 0..r_nms.data.len() {
    if r_nms.data[i] > thresh {
      descriptors.push(describe_index(&im, i));
    }
  }

  return descriptors;
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::image::{feature_normalize, load_image, same_image};
  use std::path::Path;

  #[test]
  fn test_structure() {
    let im = load_image(Path::new("images/dogbw.png")).unwrap();
    let mut s = structure_matrix(&im, 2.0);
    feature_normalize(&mut s);
    let test_im = load_image(Path::new("images/structure.png")).unwrap();
    assert!(same_image(&s, &test_im));
  }

  #[test]
  fn test_cornerness() {
    let im = load_image(Path::new("images/dogbw.png")).unwrap();
    let s = structure_matrix(&im, 2.0);
    let mut c = cornerness_response(&s);
    feature_normalize(&mut c);
    let test_im = load_image(Path::new("images/response.png")).unwrap();
    assert!(same_image(&c, &test_im));
  }
}
