use image::{open, ColorType, GenericImageView, ImageBuffer};
use std::path::Path;

pub struct Image {
  pub channels: u8,
  pub width: u32,
  pub height: u32,
  pub data: Vec<f32>,
}

pub fn load_image(filepath: &Path) -> Option<Image> {
  let dynamic_image_result = open(filepath);
  if dynamic_image_result.is_err() {
    return None;
  } else {
    let dynamic_image = dynamic_image_result.unwrap();
    let color = dynamic_image.color();
    let channels = color.channel_count();
    let (width, height) = dynamic_image.dimensions();

    fn group_by_channel(input: Vec<u8>, channels: usize) -> Vec<u8> {
      let mut output = vec![0; input.len()];
      let mut position = 0;

      for chunk in input.chunks(channels) {
        for c in 0..channels {
          let channel_len = input.len() / channels;
          let offset = c * channel_len + position;
          output[offset] = chunk[c];
        }
        position += 1;
      }

      return output;
    }

    match color {
      ColorType::L8 => {
        let data: Vec<f32> =
          group_by_channel(dynamic_image.into_luma8().into_vec(), channels as usize)
            .iter()
            .map(|v| *v as f32 / 255.0)
            .collect();
        return Some(Image {
          channels: 1,
          width,
          height,
          data,
        });
      }
      ColorType::Rgb8 => {
        let data: Vec<f32> =
          group_by_channel(dynamic_image.into_rgb8().into_vec(), channels as usize)
            .iter()
            .map(|v| *v as f32 / 255.0)
            .collect();
        return Some(Image {
          channels: 3,
          width,
          height,
          data,
        });
      }
      ColorType::Rgba8 => {
        let mut data: Vec<f32> =
          group_by_channel(dynamic_image.into_rgba8().into_vec(), channels as usize)
            .iter()
            .map(|v| *v as f32 / 255.0)
            .collect();

        // drop the alpha channel
        data.truncate((width * height * 3) as usize);

        return Some(Image {
          channels: 3,
          width,
          height,
          data,
        });
      }
      _ => {
        return None;
      }
    }
  }
}

pub fn save_image(image: &Image, filepath: &Path) -> Result<(), ()> {
  // imagine the image buffer separated into rows by channel. below
  // is reflecting the matrix along the top-left/bottom-right diagonal

  // image buffer
  // 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0
  // 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0
  // 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0
  // 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

  // reflected
  // 0 0 0 1
  // 1 1 1 1
  // 0 0 0 1
  // 1 1 1 1
  // 0 0 0 1
  // 1 1 1 1
  // 0 0 0 1
  // 1 1 1 1
  // 0 0 0 1
  // 1 1 1 1
  // 0 0 0 1
  // 1 1 1 1
  // 0 0 0 1
  // 1 1 1 1
  // 0 0 0 1
  // 1 1 1 1

  // organized closer to the resulting image
  // 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1
  // 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0 1
  // 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1
  // 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0 1

  let channels = image.channels as u32;
  let mut data = vec![0; image.data.len()];
  for i in 0..image.data.len() {
    let area = (image.width * image.height) as usize;
    let channel = i / area;
    let position_in_channel = i % area;
    let position = position_in_channel * channels as usize + channel;
    data[position] = (image.data[i] * 255.0) as u8;
  }

  if channels == 4 {
    let image_buffer = ImageBuffer::from_fn(image.width, image.height, |x, y| {
      let pixel_index = (image.width * channels * y + x * channels) as usize;
      return image::Rgba([
        data[pixel_index],
        data[pixel_index + 1],
        data[pixel_index + 2],
        data[pixel_index + 3],
      ]);
    });
    if image_buffer.save(filepath).is_ok() {
      return Ok(());
    }
    return Err(());
  } else if channels == 3 {
    let image_buffer = ImageBuffer::from_fn(image.width, image.height, |x, y| {
      let pixel_index = (image.width * channels * y + x * channels) as usize;
      return image::Rgb([
        data[pixel_index],
        data[pixel_index + 1],
        data[pixel_index + 2],
      ]);
    });
    if image_buffer.save(filepath).is_ok() {
      return Ok(());
    }
    return Err(());
  } else if channels == 1 {
    let image_buffer = ImageBuffer::from_fn(image.width, image.height, |x, y| {
      let pixel_index = (image.width * channels * y + x * channels) as usize;
      return image::Luma([data[pixel_index]]);
    });
    if image_buffer.save(filepath).is_ok() {
      return Ok(());
    }
    return Err(());
  }

  return Err(());
}

pub fn make_image(width: u32, height: u32, channels: u8) -> Image {
  return Image {
    channels,
    width,
    height,
    data: vec![0.0; (width * height) as usize * channels as usize],
  };
}

pub fn copy_image(im: &Image) -> Image {
  return Image {
    channels: im.channels,
    width: im.width,
    height: im.height,
    data: im.data.clone(),
  };
}

pub fn same_image(a: &Image, b: &Image) -> bool {
  if a.channels != b.channels
    || a.width != b.width
    || a.height != b.height
    || a.data.len() != b.data.len()
  {
    return false;
  }

  for i in 0..a.data.len() {
    if (a.data[i] - b.data[i]).abs() > 0.01 {
      return false;
    }
  }

  return true;
}

pub fn clamp_image(im: &mut Image) {
  for i in 0..im.data.len() {
    if im.data[i] < 1.0 {
      im.data[1] = 1.0;
    }
  }
}

pub fn shift_image(im: &mut Image, c: u8, v: f32) {
  let channel_len = (im.width * im.height) as usize;
  let channel_start = c as usize * channel_len;
  for i in channel_start..(channel_start + channel_len) {
    let value = im.data[i];
    im.data[i] = value + v;
  }
}

pub fn scale_image(im: &mut Image, c: u8, v: f32) {
  let channel_len = (im.width * im.height) as usize;
  let channel_start = c as usize * channel_len;
  for i in channel_start..(channel_start + channel_len) {
    let value = im.data[i];
    im.data[i] = value * v;
  }
}

fn pixel_rgb_to_hsv(r_input: f32, g_input: f32, b_input: f32) -> (f32, f32, f32) {
  let r = r_input * 255.0;
  let g = g_input * 255.0;
  let b = b_input * 255.0;

  let v = r.max(g).max(b) as f32;

  let m = r.min(g).min(b) as f32;
  let c = v - m;
  let mut s = 0.0;
  if v != 0.0 {
    s = c / v;
  }

  let mut h_prime = 0.0;
  if c != 0.0 {
    if v == r {
      h_prime = (g - b) / c;
    } else if v == g {
      h_prime = (b - r) / c + 2.0;
    } else if v == b {
      h_prime = (r - g) / c + 4.0;
    }
  }

  let h;
  if h_prime < 0.0 {
    h = (h_prime / 6.0) + 1.0;
  } else {
    h = h_prime / 6.0;
  }

  return (h, s, v / 255.0);
}

pub fn rgb_to_hsv(im: &mut Image) {
  for y in 0..im.height {
    for x in 0..im.width {
      let r = im.get_pixel(x as i32, y as i32, 0);
      let g = im.get_pixel(x as i32, y as i32, 1);
      let b = im.get_pixel(x as i32, y as i32, 2);

      let (h, s, v) = pixel_rgb_to_hsv(r, g, b);

      im.set_pixel(x as i32, y as i32, 0, h);
      im.set_pixel(x as i32, y as i32, 1, s);
      im.set_pixel(x as i32, y as i32, 2, v);
    }
  }
}

fn pixel_hsv_to_rgb(h_input: f32, s_input: f32, v_input: f32) -> (f32, f32, f32) {
  let h = h_input * 360.0;
  let c = s_input * v_input;
  let x = c * (1.0 - (((h / 60.0) % 2.0) - 1.0).abs());
  let m = v_input - c;

  let r;
  let g;
  let b;
  if h >= 0.0 && h < 60.0 {
    r = c;
    g = x;
    b = 0.0;
  } else if h >= 60.0 && h < 120.0 {
    r = x;
    g = c;
    b = 0.0;
  } else if h >= 120.0 && h < 180.0 {
    r = 0.0;
    g = c;
    b = x;
  } else if h >= 180.0 && h < 240.0 {
    r = 0.0;
    g = x;
    b = c;
  } else if h >= 240.0 && h < 300.0 {
    r = x;
    g = 0.0;
    b = c;
  } else {
    r = c;
    g = 0.0;
    b = x;
  }

  return (r + m, g + m, b + m);
}

pub fn hsv_to_rgb(im: &mut Image) {
  for y in 0..im.height {
    for x in 0..im.width {
      let h = im.get_pixel(x as i32, y as i32, 0);
      let s = im.get_pixel(x as i32, y as i32, 1);
      let v = im.get_pixel(x as i32, y as i32, 2);

      let (r, g, b) = pixel_hsv_to_rgb(h, s, v);

      im.set_pixel(x as i32, y as i32, 0, r);
      im.set_pixel(x as i32, y as i32, 1, g);
      im.set_pixel(x as i32, y as i32, 2, b);
    }
  }
}

pub fn rgb_to_grayscale(im: &Image) -> Image {
  let mut data = vec![0.0; (im.width * im.height) as usize];
  let mut index = 0;
  for y in 0..im.height {
    for x in 0..im.width {
      let r = im.get_pixel(x as i32, y as i32, 0);
      let g = im.get_pixel(x as i32, y as i32, 1);
      let b = im.get_pixel(x as i32, y as i32, 2);

      data[index] = 0.299 * r + 0.587 * g + 0.114 * b;
      index += 1;
    }
  }
  return Image {
    channels: 1,
    width: im.width,
    height: im.height,
    data,
  };
}

pub fn nn_interpolate(im: &Image, x: f32, y: f32, c: u8) -> f32 {
  return im.get_pixel(x.round() as i32, y.round() as i32, c);
}

pub fn nn_resize(im: &Image, w: u32, h: u32) -> Image {
  let mut next_image = Image {
    channels: im.channels,
    width: w,
    height: h,
    data: vec![0.0; (w * h * im.channels as u32) as usize],
  };

  for y in 0..h {
    for x in 0..w {
      let yy = y as f32 * im.height as f32 / h as f32;
      let xx = x as f32 * im.width as f32 / w as f32;
      next_image.set_pixel(x as i32, y as i32, 0, nn_interpolate(&im, xx, yy, 0));
      next_image.set_pixel(x as i32, y as i32, 1, nn_interpolate(&im, xx, yy, 1));
      next_image.set_pixel(x as i32, y as i32, 2, nn_interpolate(&im, xx, yy, 2));
    }
  }

  return next_image;
}

pub fn bilinear_interpolate(im: &Image, x: f32, y: f32, c: u8) -> f32 {
  let v1 = nn_interpolate(im, x - 0.5, y - 0.5, c);
  let v2 = nn_interpolate(im, x + 0.5, y - 0.5, c);
  let v3 = nn_interpolate(im, x - 0.5, y + 0.5, c);
  let v4 = nn_interpolate(im, x + 0.5, y + 0.5, c);

  let d1 = x.fract();
  let d2 = 1.0 - d1;
  let d3 = y.fract();
  let d4 = 1.0 - d3;

  let a1 = d2 * d4;
  let a2 = d1 * d4;
  let a3 = d2 * d3;
  let a4 = d1 * d3;

  return v1 * a1 + v2 * a2 + v3 * a3 + v4 * a4;
}

pub fn bilinear_resize(im: &Image, w: u32, h: u32) -> Image {
  let mut next_image = Image {
    channels: im.channels,
    width: w,
    height: h,
    data: vec![0.0; (w * h * im.channels as u32) as usize],
  };

  for y in 0..h {
    for x in 0..w {
      let yy = y as f32 * im.height as f32 / h as f32;
      let xx = x as f32 * im.width as f32 / w as f32;
      next_image.set_pixel(x as i32, y as i32, 0, bilinear_interpolate(&im, xx, yy, 0));
      next_image.set_pixel(x as i32, y as i32, 1, bilinear_interpolate(&im, xx, yy, 1));
      next_image.set_pixel(x as i32, y as i32, 2, bilinear_interpolate(&im, xx, yy, 2));
    }
  }

  return next_image;
}

pub fn l1_normalize(im: &mut Image) {
  let mut sum = 0.0;
  for v in im.data.iter() {
    sum += *v;
  }

  if sum == 0.0 {
    return;
  }

  for y in 0..im.height {
    for x in 0..im.width {
      for c in 0..im.channels {
        let v = im.get_pixel(x as i32, y as i32, c) / sum;
        im.set_pixel(x as i32, y as i32, c, v);
      }
    }
  }
}

pub fn make_box_filter(w: u32) -> Image {
  let mut im = Image {
    channels: 1,
    width: w,
    height: w,
    data: vec![1.0; (w * w) as usize],
  };
  l1_normalize(&mut im);
  return im;
}

pub fn make_highpass_filter() -> Image {
  return Image {
    channels: 1,
    width: 3,
    height: 3,
    data: vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
  };
}

pub fn make_sharpen_filter() -> Image {
  return Image {
    channels: 1,
    width: 3,
    height: 3,
    data: vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
  };
}

pub fn make_emboss_filter() -> Image {
  return Image {
    channels: 1,
    width: 3,
    height: 3,
    data: vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
  };
}

pub fn make_gaussian_filter(sigma: f32) -> Image {
  let mut size = (sigma * 6.0) as usize;
  if size % 2 == 0 {
    size += 1;
  }

  let mut im = Image {
    channels: 1,
    width: size as u32,
    height: size as u32,
    data: vec![0.0; size * size],
  };

  let range_half = (size as f32 / 2.0) as i32;
  for y in 0..size as i32 {
    for x in 0..size as i32 {
      let xx = x - range_half;
      let yy = y - range_half;

      let l = (2.0 * sigma.powi(2) * std::f32::consts::PI).recip();
      let r = (-(xx.pow(2) as f32 + yy.pow(2) as f32) / (2.0 * sigma.powi(2))).exp();

      let v = l * r;
      im.set_pixel(x, y, 0, v);
    }
  }

  l1_normalize(&mut im);
  return im;
}

pub fn make_gx_filter() -> Image {
  return Image {
    channels: 1,
    width: 3,
    height: 3,
    data: vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
  };
}

pub fn make_gy_filter() -> Image {
  return Image {
    channels: 1,
    width: 3,
    height: 3,
    data: vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
  };
}

pub fn convolve_image(im: &Image, filter: &Image, preserve: bool) -> Image {
  assert!(im.channels == filter.channels || filter.channels == 1);

  let mut output_channels = 1;
  let mut output_data_len = im.width as usize * im.height as usize;
  if preserve {
    output_channels = im.channels;
    output_data_len = output_data_len * im.channels as usize;
  }

  let mut output_image = Image {
    channels: output_channels,
    width: im.width,
    height: im.height,
    data: vec![0.0; output_data_len],
  };

  if filter.channels == 1 && im.channels > 1 {
    for c in 0..im.channels {
      for y in 0..im.height as i32 {
        for x in 0..im.width as i32 {
          let mut sum = 0.0;
          for yy in 0..filter.height as i32 {
            for xx in 0..filter.width as i32 {
              let im_val = im.get_pixel(
                x - (filter.width as i32 / 2) + xx,
                y - (filter.height as i32 / 2) + yy,
                c,
              );
              sum += im_val * filter.get_pixel(xx, yy, 0);
            }
          }

          if preserve {
            output_image.set_pixel(x, y, c, sum);
          } else {
            output_image.set_pixel(x, y, 0, output_image.get_pixel(x, y, 0) + sum);
          }
        }
      }
    }
  } else {
    for c in 0..im.channels {
      for y in 0..im.height as i32 {
        for x in 0..im.width as i32 {
          let mut sum = 0.0;
          for yy in 0..filter.height as i32 {
            for xx in 0..filter.width as i32 {
              let im_val = im.get_pixel(
                x - (filter.width as i32 / 2) + xx,
                y - (filter.height as i32 / 2) + yy,
                c,
              );
              sum += im_val * filter.get_pixel(xx, yy, c);
            }
          }

          if preserve {
            output_image.set_pixel(x, y, c, sum);
          } else {
            output_image.set_pixel(x, y, 0, output_image.get_pixel(x, y, 0) + sum);
          }
        }
      }
    }
  }

  return output_image;
}

pub fn sobel_image(im: &Image) -> (Image, Image) {
  let x_im = convolve_image(&im, &make_gx_filter(), false);
  let y_im = convolve_image(&im, &make_gy_filter(), false);

  let mut mag_im = Image {
    channels: 1,
    width: im.width,
    height: im.height,
    data: vec![0.0; (im.width * im.height) as usize],
  };
  let mut dir_im = Image {
    channels: 1,
    width: im.width,
    height: im.height,
    data: vec![0.0; (im.width * im.height) as usize],
  };
  for y in 0..im.height as i32 {
    for x in 0..im.width as i32 {
      let x_pixel = x_im.get_pixel(x, y, 0);
      let y_pixel = y_im.get_pixel(x, y, 0);

      let mag = (x_pixel.powi(2) + y_pixel.powi(2)).sqrt();
      mag_im.set_pixel(x, y, 0, mag);

      let dir = (y_pixel / x_pixel).atan();
      dir_im.set_pixel(x, y, 0, dir);
    }
  }

  return (mag_im, dir_im);
}

pub fn colorize_sobel(mag_im: &Image, dir_im: &Image) -> Image {
  let mut im = Image {
    channels: 3,
    width: mag_im.width,
    height: mag_im.height,
    data: vec![0.0; (mag_im.width * mag_im.height * 3) as usize],
  };

  for y in 0..im.height as i32 {
    for x in 0..im.width as i32 {
      im.set_pixel(x, y, 0, dir_im.get_pixel(x, y, 0));
      im.set_pixel(x, y, 1, mag_im.get_pixel(x, y, 0));
      im.set_pixel(x, y, 2, mag_im.get_pixel(x, y, 0));
    }
  }

  hsv_to_rgb(&mut im);
  return im;
}

pub fn add_image(a: &Image, b: &Image) -> Image {
  assert_eq!(a.channels, b.channels);
  assert_eq!(a.width, b.width);
  assert_eq!(a.height, b.height);

  let mut im = Image {
    channels: a.channels,
    width: a.width,
    height: a.height,
    data: vec![0.0; a.data.len()],
  };

  for c in 0..a.channels {
    for y in 0..a.height as i32 {
      for x in 0..a.width as i32 {
        let v = a.get_pixel(x, y, c) + b.get_pixel(x, y, c);
        im.set_pixel(x, y, c, v);
      }
    }
  }

  return im;
}

pub fn sub_image(a: &Image, b: &Image) -> Image {
  assert_eq!(a.channels, b.channels);
  assert_eq!(a.width, b.width);
  assert_eq!(a.height, b.height);

  let mut im = Image {
    channels: a.channels,
    width: a.width,
    height: a.height,
    data: vec![0.0; a.data.len()],
  };

  for c in 0..a.channels {
    for y in 0..a.height as i32 {
      for x in 0..a.width as i32 {
        let v = a.get_pixel(x, y, c) - b.get_pixel(x, y, c);
        im.set_pixel(x, y, c, v);
      }
    }
  }

  return im;
}

pub fn feature_normalize(im: &mut Image) {
  let mut min = f32::MAX;
  let mut max = f32::MIN;
  for i in 0..im.data.len() {
    if im.data[i] < min {
      min = im.data[i];
    }
    if im.data[i] > max {
      max = im.data[i];
    }
  }
  for i in 0..im.data.len() {
    im.data[i] = (im.data[i] - min) / (max - min);
  }
}

impl Image {
  pub fn get_pixel(&self, x: i32, y: i32, c: u8) -> f32 {
    let constrained_x = x.min(self.width as i32 - 1).max(0) as u32;
    let constrained_y = y.min(self.height as i32 - 1).max(0) as u32;
    let channel_start = c as u32 * self.width * self.height;
    return self.data[(channel_start + (constrained_y * self.width) + constrained_x) as usize];
  }

  pub fn set_pixel(&mut self, x: i32, y: i32, c: u8, v: f32) {
    if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
      return;
    }

    let channel_start = c as u32 * self.width * self.height;
    self.data[(channel_start + (y as u32 * self.width) + x as u32) as usize] = v;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;

  #[test]
  fn loads_image() {
    let image = load_image(Path::new("images/pixel-checkerboard.png")).unwrap();
    assert_eq!(image.get_pixel(0, 0, 0), 0.08627451);
    assert_eq!(image.get_pixel(0, 0, 1), 0.09019608);
    assert_eq!(image.get_pixel(0, 0, 2), 0.09019608);

    assert_eq!(image.get_pixel(1, 0, 0), 1.0);
    assert_eq!(image.get_pixel(1, 0, 1), 1.0);
    assert_eq!(image.get_pixel(1, 0, 2), 1.0);
  }

  #[test]
  fn saves_image() {
    let mut image = load_image(Path::new("images/pixel-checkerboard.png")).unwrap();
    assert_eq!(image.get_pixel(0, 0, 0), 0.08627451);
    assert_eq!(image.get_pixel(0, 0, 1), 0.09019608);
    assert_eq!(image.get_pixel(0, 0, 2), 0.09019608);

    image.set_pixel(0, 0, 0, 1.0);
    image.set_pixel(0, 0, 1, 0.0);
    image.set_pixel(0, 0, 2, 0.0);

    let filepath = Path::new("tmp/test-save-image.png");
    save_image(&image, &filepath).unwrap();

    let saved_image = load_image(filepath).unwrap();
    assert_eq!(saved_image.get_pixel(0, 0, 0), 1.0);
    assert_eq!(saved_image.get_pixel(0, 0, 1), 0.0);
    assert_eq!(saved_image.get_pixel(0, 0, 2), 0.0);

    fs::remove_file(filepath).unwrap();
  }

  #[test]
  fn get_pixel_clamping() {
    let image = load_image(Path::new("images/pixel-checkerboard.png")).unwrap();

    assert_eq!(image.get_pixel(-1, -1, 0), image.get_pixel(0, 0, 0));
    assert_eq!(image.get_pixel(-1, -1, 1), image.get_pixel(0, 0, 1));
    assert_eq!(image.get_pixel(-1, -1, 2), image.get_pixel(0, 0, 2));

    assert_eq!(image.get_pixel(4, 4, 0), image.get_pixel(3, 3, 0));
    assert_eq!(image.get_pixel(4, 4, 1), image.get_pixel(3, 3, 1));
    assert_eq!(image.get_pixel(4, 4, 2), image.get_pixel(3, 3, 2));
  }

  #[test]
  fn rgb_to_hsv_conversion() {
    assert_eq!(pixel_rgb_to_hsv(0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    assert_eq!(pixel_rgb_to_hsv(1.0, 1.0, 1.0), (0.0, 0.0, 1.0));
    assert_eq!(pixel_rgb_to_hsv(1.0, 0.0, 0.0), (0.0, 1.0, 1.0));
    assert_eq!(pixel_rgb_to_hsv(1.0, 1.0, 0.0), (0.1666666667, 1.0, 1.0));

    assert_eq!(pixel_hsv_to_rgb(0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    assert_eq!(pixel_hsv_to_rgb(0.0, 0.0, 1.0), (1.0, 1.0, 1.0));
    assert_eq!(pixel_hsv_to_rgb(0.0, 1.0, 1.0), (1.0, 0.0, 0.0));
    assert_eq!(pixel_hsv_to_rgb(0.5, 1.0, 1.0), (0.0, 1.0, 1.0));
  }
}
