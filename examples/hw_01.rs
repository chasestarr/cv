use cv::image::{
  clamp_image, hsv_to_rgb, load_image, rgb_to_grayscale, rgb_to_hsv, save_image, scale_image,
  shift_image,
};
use std::path::Path;

fn remove_red() {
  let mut im = load_image(Path::new("images/dog.jpg")).unwrap();
  for x in 0..im.width {
    for y in 0..im.height {
      im.set_pixel(x as i32, y as i32, 0, 0.0);
    }
  }
  save_image(&im, Path::new("tmp/hw_01_dog_no_red.jpg")).unwrap();
}

fn grayscale() {
  let im = load_image(Path::new("images/colorbar.png")).unwrap();
  let gray = rgb_to_grayscale(&im);
  save_image(&gray, Path::new("tmp/hw_01_colorbar_gray.jpg")).unwrap();
}

fn shift() {
  let mut im = load_image(Path::new("images/dog.jpg")).unwrap();
  shift_image(&mut im, 0, 0.4);
  shift_image(&mut im, 1, 0.4);
  shift_image(&mut im, 2, 0.4);
  clamp_image(&mut im);
  save_image(&im, Path::new("tmp/hw_01_shift.jpg")).unwrap();
}

fn hsv() {
  let mut im = load_image(Path::new("images/dog.jpg")).unwrap();
  rgb_to_hsv(&mut im);
  scale_image(&mut im, 1, 2.0);
  clamp_image(&mut im);
  hsv_to_rgb(&mut im);
  save_image(&im, Path::new("tmp/hw_01_hsv.jpg")).unwrap();
}

fn main() {
  remove_red();
  grayscale();
  shift();
  hsv();
}
