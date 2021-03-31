use cv::image::{
  add_image, colorize_sobel, convolve_image, feature_normalize, load_image, make_box_filter,
  make_emboss_filter, make_gaussian_filter, make_gx_filter, make_gy_filter, make_highpass_filter,
  make_sharpen_filter, save_image, sobel_image, sub_image,
};
use std::path::Path;

fn filters() {
  let im = load_image(Path::new("images/dog.jpg")).unwrap();

  let box_filter = make_box_filter(7);
  let box_im = convolve_image(&im, &box_filter, true);
  save_image(&box_im, Path::new("tmp/hw_03_box.jpg")).unwrap();

  let highpass_filter = make_highpass_filter();
  let highpass = convolve_image(&im, &highpass_filter, true);
  save_image(&highpass, Path::new("tmp/hw_03_highpass.jpg")).unwrap();

  let sharpen_filter = make_sharpen_filter();
  let sharpen = convolve_image(&im, &sharpen_filter, true);
  save_image(&sharpen, Path::new("tmp/hw_03_sharpen.jpg")).unwrap();

  let emboss_filter = make_emboss_filter();
  let emboss = convolve_image(&im, &emboss_filter, true);
  save_image(&emboss, Path::new("tmp/hw_03_emboss.jpg")).unwrap();

  let gaussian_filter = make_gaussian_filter(2.0);
  let gaussian = convolve_image(&im, &gaussian_filter, true);
  save_image(&gaussian, Path::new("tmp/hw_03_gaussian.jpg")).unwrap();
}

fn arithmetic() {
  let im = load_image(Path::new("images/dog.jpg")).unwrap();
  let gaussian_filter = make_gaussian_filter(4.0);
  let low_freq = convolve_image(&im, &gaussian_filter, true);
  save_image(&low_freq, Path::new("tmp/hw_03_low_freq.jpg")).unwrap();
  let high_freq = sub_image(&im, &low_freq);
  save_image(&high_freq, Path::new("tmp/hw_03_high_freq.jpg")).unwrap();
  let reconstruct = add_image(&high_freq, &low_freq);
  save_image(&reconstruct, Path::new("tmp/hw_03_reconstruct.jpg")).unwrap();
}

fn sobel() {
  let im = load_image(Path::new("images/valve.png")).unwrap();

  let x_filter = make_gx_filter();
  let x_im = convolve_image(&im, &x_filter, false);
  save_image(&x_im, Path::new("tmp/hw_03_sobel_x.jpg")).unwrap();

  let y_filter = make_gy_filter();
  let y_im = convolve_image(&im, &y_filter, false);
  save_image(&y_im, Path::new("tmp/hw_03_sobel_y.jpg")).unwrap();

  let mut xy_im = add_image(&x_im, &y_im);
  feature_normalize(&mut xy_im);
  save_image(&xy_im, Path::new("tmp/hw_03_sobel_xy.jpg")).unwrap();

  let (mut mag, mut dir) = sobel_image(&im);
  feature_normalize(&mut mag);
  feature_normalize(&mut dir);
  save_image(&mag, Path::new("tmp/hw_03_sobel_mag.jpg")).unwrap();

  let colorized = colorize_sobel(&mag, &dir);
  save_image(&colorized, Path::new("tmp/hw_03_sobel_colorized.jpg")).unwrap();
}

fn main() {
  filters();
  arithmetic();
  sobel();
}
